"""
Ghost Member API 서버 — 파일이 들어오면 자동으로 파싱→임베딩→Qdrant 저장
====================================================

실행 방법:
    source ~/venv312/bin/activate
    pip install fastapi uvicorn python-multipart qdrant-client ibm-watsonx-ai

    export WATSONX_API_KEY="..."
    export WATSONX_PROJECT_ID="..."
    export QDRANT_URL="http://localhost:6333"     # 도커: docker run -p 6333:6333 qdrant/qdrant

    uvicorn app:app --reload
    → 브라우저에서 http://localhost:8000/docs 열면
      스웨거 UI에서 파일 업로드까지 바로 테스트 가능!

엔드포인트 구성:
    POST /upload          md/ts/tsx 파일 업로드 → 파싱 → 임베딩 → Qdrant 저장
    POST /connect-github  깃허브 저장소 연동 → 커밋+PR 수집 → 임베딩 → 저장
    POST /ask             질문 → 벡터 검색 → LLM 답변 (RAG)
    GET  /health          서버 살아있는지 확인
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from parsers import parse_markdown, parse_ts, fetch_github_data
from normalizer import (
    normalize_markdown,
    normalize_code,
    normalize_git,
    build_embedding_text,
)

# ===========================================================================
# 설정값 (팀에서 통일해야 하는 것들)
# ===========================================================================
COLLECTION = "ghost_member"
EMBED_MODEL = "ibm/granite-embedding-278m-multilingual"  # 한국어 지원 임베딩 모델
EMBED_DIM = 768          # 위 모델의 벡터 차원 — 바꾸면 컬렉션도 다시 만들어야 함!
LLM_MODEL = "ibm/granite-3-8b-instruct"
TOP_K = 5                # 질문당 검색할 근거 청크 수

app = FastAPI(title="Ghost Member API", description="팀 기여도 분석 RAG 백엔드")


# ===========================================================================
# 클라이언트 준비 (lazy 초기화)
# ===========================================================================
# 왜 lazy(처음 쓸 때 생성)로 하나?
#   서버 시작 시점에 만들면 API 키가 없을 때 서버 자체가 안 떠서
#   /docs 확인조차 못 한다. 처음 요청이 올 때 만들고 이후 재사용한다.
_embedder = None
_qdrant = None


def get_embedder():
    """watsonx 임베딩 클라이언트 (최초 1회만 생성 후 재사용)"""
    global _embedder
    if _embedder is None:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import Embeddings

        _embedder = Embeddings(
            model_id=EMBED_MODEL,
            credentials=Credentials(
                url=os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
                api_key=os.environ["WATSONX_API_KEY"],
            ),
            project_id=os.environ["WATSONX_PROJECT_ID"],
        )
    return _embedder


def get_qdrant():
    """Qdrant 클라이언트 + 컬렉션 보장 (없으면 생성)"""
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        _qdrant = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        if not _qdrant.collection_exists(COLLECTION):
            _qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
    return _qdrant


# ===========================================================================
# 공통 처리: 청크 리스트 → 임베딩 → Qdrant 저장
#   모든 엔드포인트가 마지막에 이 함수를 호출한다.
# ===========================================================================
def index_chunks(chunks: list[dict]) -> int:
    """공통 청크들을 임베딩해서 Qdrant에 upsert. 저장된 개수를 반환."""
    if not chunks:
        return 0

    from qdrant_client.models import PointStruct

    embedder = get_embedder()
    client = get_qdrant()

    BATCH = 50  # 임베딩 API 한 번에 보낼 텍스트 수 (한도 보호)
    saved = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [build_embedding_text(c) for c in batch]
        vectors = embedder.embed_documents(texts=texts)

        points = [
            PointStruct(
                id=c.get("chunk_id") or str(uuid.uuid4()),
                vector=v,
                # payload = 검색 시 돌려받을 정보 + 필터 축(source_type, author)
                payload={
                    "source_type": c["source_type"],
                    "title": c["title"],
                    "content": c["content"],
                    "author": c.get("author"),
                    "created_at": c.get("created_at"),
                    "file_path": c.get("file_path"),
                    **c.get("metadata", {}),
                },
            )
            for c, v in zip(batch, vectors)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        saved += len(points)
    return saved


# ===========================================================================
# 엔드포인트 1: 파일 업로드
# ===========================================================================
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    author: str | None = Form(None),  # 업로드한 팀원 이름 (코드 파일 기여자 표시용)
):
    """
    md / ts / tsx 파일을 받아 즉시 파싱→임베딩→저장한다.

    프론트에서는 <input type="file"> + FormData로 보내면 됨:
        const fd = new FormData();
        fd.append("file", 선택한파일);
        fd.append("author", "kim");
        fetch("/upload", { method: "POST", body: fd });
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".md", ".ts", ".tsx"):
        raise HTTPException(400, f"지원하지 않는 파일 형식: {suffix} (md/ts/tsx만 가능)")

    # UploadFile은 메모리/스트림이라 경로 기반인 파서에 바로 못 넣는다.
    # → 임시 파일로 저장했다가 파싱 후 삭제한다.
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 확장자를 보고 어떤 파서를 쓸지 분기 — 여기가 "자동 파싱기"의 심장
        if suffix == ".md":
            chunks = normalize_markdown(parse_markdown(tmp_path), file.filename)
        else:  # .ts / .tsx
            chunks = normalize_code(parse_ts(tmp_path), file.filename, author=author)
    except Exception as e:
        raise HTTPException(422, f"파싱 실패: {e}")
    finally:
        os.unlink(tmp_path)  # 임시 파일 정리 (성공/실패 무관)

    saved = index_chunks(chunks)
    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "chunks_indexed": saved,
        # 프론트에서 "어떻게 잘렸는지" 보여줄 수 있게 제목 목록도 반환
        "titles": [c["title"] for c in chunks],
    }


# ===========================================================================
# 엔드포인트 1-B: zip 업로드 (프로젝트 폴더 통째로)
# ===========================================================================
@app.post("/upload-zip")
async def upload_zip(
    file: UploadFile = File(...),
    author: str | None = Form(None),  # 업로드한 팀원 이름
):
    """
    프로젝트 폴더를 zip으로 받아 전체를 파싱→임베딩→저장한다.

    - 내부의 모든 .md / .ts / .tsx를 자동으로 파싱
    - zip 안에 .git 폴더가 포함돼 있으면 커밋 히스토리까지 파싱
    - node_modules, __MACOSX 등 불필요한 폴더는 자동 제외

    프론트에서 "폴더 업로드"를 구현하는 방법:
      (권장) 사용자가 폴더를 zip으로 압축해서 올리게 안내
      (고급) <input webkitdirectory>로 폴더 선택 → JSZip으로 브라우저에서
             zip으로 묶어 이 엔드포인트로 전송
    """
    import shutil
    import zipfile

    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "zip 파일만 업로드할 수 있어요")

    content = await file.read()

    # 임시 폴더에 zip 저장 → 압축 해제 → 폴더 파싱 → 폴더 삭제
    tmp_dir = tempfile.mkdtemp(prefix="ghost_upload_")
    try:
        zip_path = os.path.join(tmp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(content)

        extract_dir = os.path.join(tmp_dir, "extracted")
        try:
            # extractall은 절대경로/.. 을 제거해주므로 zip 경로조작 공격에 안전
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(422, "손상된 zip 파일이에요")

        # 압축을 풀면 보통 "폴더 하나"가 나온다 (프로젝트루트/src/...).
        # 그 경우 안쪽 폴더를 루트로 삼아야 상대경로가 깔끔해진다.
        entries = [p for p in Path(extract_dir).iterdir() if p.name != "__MACOSX"]
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(extract_dir)

        # ingest.py의 폴더 순회 로직 재사용:
        # md/ts/tsx 전체 파싱 + .git 있으면 커밋 히스토리까지
        from ingest import ingest_project
        chunks = ingest_project(root, author=author)

        saved = index_chunks(chunks)

        # 소스 타입별 개수 요약 (프론트에서 "문서 24개, 코드 21개..." 표시용)
        counts: dict[str, int] = {}
        for c in chunks:
            counts[c["source_type"]] = counts.get(c["source_type"], 0) + 1

        return {
            "filename": file.filename,
            "chunks_created": len(chunks),
            "chunks_indexed": saved,
            "by_source_type": counts,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # 성공/실패 무관 임시 폴더 정리


# ===========================================================================
# 엔드포인트 2: 깃허브 연동
# ===========================================================================
class GithubRequest(BaseModel):
    owner: str                     # 예: "ghost-member-team"
    repo: str                      # 예: "backend"
    token: str | None = None       # GitHub PAT (공개 저장소면 생략 가능하나 권장)


@app.post("/connect-github")
def connect_github(req: GithubRequest):
    """
    깃허브 저장소의 커밋 + PR(필수)을 수집해 임베딩→저장한다.
    저장소가 크면 시간이 걸리므로, 프론트에서 로딩 표시를 해줄 것.
    """
    try:
        git_data = fetch_github_data(req.owner, req.repo, req.token)
    except Exception as e:
        raise HTTPException(502, f"GitHub API 호출 실패: {e}")

    chunks = normalize_git(git_data)
    saved = index_chunks(chunks)
    return {
        "repo": f"{req.owner}/{req.repo}",
        "commits": len(git_data["commits"]),
        "pull_requests": len(git_data["pull_requests"]),
        "chunks_indexed": saved,
    }


# ===========================================================================
# 엔드포인트 3: RAG 질문
# ===========================================================================
class AskRequest(BaseModel):
    question: str
    source_type: str | None = None  # "pull_request" 등으로 검색 범위 좁히기 (선택)


@app.post("/ask")
def ask(req: AskRequest):
    """질문 → 벡터 검색 → 근거와 함께 LLM 답변 생성"""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    embedder = get_embedder()
    client = get_qdrant()

    # 1) 질문도 같은 모델로 벡터화 (문서와 같은 벡터 공간에 놓기 위해)
    q_vector = embedder.embed_query(text=req.question)

    # 2) 선택적 필터: source_type이 지정되면 그 종류 안에서만 검색
    q_filter = None
    if req.source_type:
        q_filter = Filter(must=[
            FieldCondition(key="source_type", match=MatchValue(value=req.source_type))
        ])

    hits = client.query_points(
        collection_name=COLLECTION,
        query=q_vector,
        query_filter=q_filter,
        limit=TOP_K,
        with_payload=True,
    ).points

    if not hits:
        return {"answer": "관련 자료를 찾지 못했어요. 먼저 문서/코드를 업로드해 주세요.", "sources": []}

    # 3) 검색 결과를 근거 목록으로 조립 → LLM에게 "근거만 갖고 답하라"고 지시
    context = "\n\n".join(
        f"[{i}] ({h.payload['source_type']}) {h.payload['title']}\n{h.payload['content'][:500]}"
        for i, h in enumerate(hits, 1)
    )
    prompt = f"""당신은 팀 프로젝트 기여도 분석 도우미입니다.
아래 [근거 자료]만 사용해서 질문에 답하세요.
근거에 없는 내용은 지어내지 말고 "자료에 없음"이라고 답하세요.
답변에 사용한 근거 번호를 [1]처럼 표시하세요.

[근거 자료]
{context}

[질문]
{req.question}

[답변]"""

    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    model = ModelInference(
        model_id=LLM_MODEL,
        credentials=Credentials(
            url=os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            api_key=os.environ["WATSONX_API_KEY"],
        ),
        project_id=os.environ["WATSONX_PROJECT_ID"],
    )
    answer_text = model.generate_text(prompt=prompt, params={"max_new_tokens": 500})

    return {
        "answer": answer_text,
        # 프론트에서 "근거 보기"를 만들 수 있게 출처도 함께 반환
        "sources": [
            {
                "rank": i,
                "score": round(h.score, 3),
                "source_type": h.payload["source_type"],
                "title": h.payload["title"],
                "author": h.payload.get("author"),
            }
            for i, h in enumerate(hits, 1)
        ],
    }


@app.get("/health")
def health():
    """배포 후 살아있는지 확인용 (Code Engine 헬스체크에도 사용 가능)"""
    return {"status": "ok"}
