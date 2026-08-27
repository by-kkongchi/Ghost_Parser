"""
전체 파이프라인 예제 — 파싱 → 노멀라이즈 → Qdrant 업로드
====================================================

실행 전 준비:
  pip install qdrant-client ibm-watsonx-ai   # (임베딩을 watsonx로 할 경우)

이 파일은 "붙이는 방법"을 보여주는 예제이므로,
실제 서비스에서는 FastAPI 엔드포인트 안에서 같은 흐름을 호출하면 된다.
  - POST /upload/doc   → parse_markdown + normalize_markdown
  - POST /upload/code  → parse_ts + normalize_code
  - POST /connect/github → fetch_github_data + normalize_git
"""

from parsers import parse_markdown, parse_ts, fetch_github_data
from normalizer import (
    normalize_markdown,
    normalize_code,
    normalize_git,
    to_qdrant_points,
)

# ---------------------------------------------------------------------------
# 1. 각 소스 파싱
# ---------------------------------------------------------------------------

# ① 요구사항 설계 문서 (heading 기준 청킹)
md_chunks = parse_markdown("sample_data/설계문서.md")

# ② 개발 코드 (최상위 선언 기준 청킹)
code_chunks = parse_ts("sample_data/Login.tsx")

# ③ 깃허브 연동 (커밋 + PR — PR은 필수!)
#    토큰은 환경변수로 관리 권장: os.environ["GITHUB_TOKEN"]
# git_data = fetch_github_data("팀오너", "저장소이름", token="ghp_...")
git_data = {"commits": [], "pull_requests": []}  # 데모용 빈 데이터

# ---------------------------------------------------------------------------
# 2. 공통 스키마로 노멀라이즈
# ---------------------------------------------------------------------------
chunks = (
    normalize_markdown(md_chunks, "sample_data/설계문서.md")
    + normalize_code(code_chunks, "sample_data/Login.tsx", author="업로더이름")
    + normalize_git(git_data)
)
print(f"총 {len(chunks)}개 청크 생성")

# ---------------------------------------------------------------------------
# 3. 임베딩 함수 정의 (아래 중 하나 선택)
# ---------------------------------------------------------------------------

# (A) watsonx.ai 임베딩을 쓰는 경우:
#
# from ibm_watsonx_ai import Credentials
# from ibm_watsonx_ai.foundation_models import Embeddings
#
# embedder = Embeddings(
#     model_id="ibm/granite-embedding-278m-multilingual",  # 한국어 지원 모델
#     credentials=Credentials(url="https://us-south.ml.cloud.ibm.com", api_key="..."),
#     project_id="...",
# )
# def embed_fn(texts):
#     return embedder.embed_documents(texts=texts)

# (B) 데모용 더미 임베딩 (Qdrant 연결 없이 구조만 확인할 때):
def embed_fn(texts):
    return [[0.0] * 768 for _ in texts]

# ---------------------------------------------------------------------------
# 4. Qdrant 포인트로 변환 후 업로드
# ---------------------------------------------------------------------------
points = to_qdrant_points(chunks, embed_fn)

# from qdrant_client import QdrantClient
# from qdrant_client.models import PointStruct, VectorParams, Distance
#
# client = QdrantClient(url="http://localhost:6333")
#
# # 컬렉션은 최초 1회만 생성 (벡터 차원은 임베딩 모델에 맞출 것!)
# client.recreate_collection(
#     collection_name="ghost_member",
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE),
# )
# client.upsert(
#     collection_name="ghost_member",
#     points=[PointStruct(**p) for p in points],
# )

print(f"Qdrant 포인트 {len(points)}개 준비 완료")
if points:
    sample = {**points[0], "vector": f"<{len(points[0]['vector'])}차원 벡터>"}
    import json
    print(json.dumps(sample, ensure_ascii=False, indent=2)[:600])
