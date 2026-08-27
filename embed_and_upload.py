"""
⑤ 임베딩 + Qdrant 업로드 — RAG 인덱싱의 마지막 단계
====================================================

사용법:
    # 1) 환경변수 설정 (팀 watsonx 계정 정보)
    export WATSONX_API_KEY="..."
    export WATSONX_PROJECT_ID="..."
    export QDRANT_URL="http://localhost:6333"      # 또는 Qdrant Cloud URL
    # export QDRANT_API_KEY="..."                  # Qdrant Cloud면 필요

    # 2) ingest.py가 만든 chunks JSON을 업로드
    python embed_and_upload.py sample_data/chunks_frontend_handover_employee_sample.json

사전 설치:
    pip install qdrant-client ibm-watsonx-ai

로컬 Qdrant 띄우기 (도커 없이는 Qdrant Cloud 무료 티어 추천):
    docker run -p 6333:6333 qdrant/qdrant
    → 도커가 없다면 https://cloud.qdrant.io 무료 클러스터 생성 후 URL/API키 사용
"""

from __future__ import annotations

import json
import os
import sys
import uuid

COLLECTION = "ghost_member"           # Qdrant 컬렉션 이름 (팀에서 통일할 것)
EMBED_MODEL = "ibm/granite-embedding-278m-multilingual"  # 한국어 되는 watsonx 임베딩 모델
EMBED_DIM = 768                       # 위 모델의 벡터 차원 — 컬렉션 생성 시 일치 필수!
BATCH = 50                            # 한 번에 임베딩할 청크 수 (API 한도 보호)


def get_embedder():
    """watsonx.ai 임베딩 클라이언트 생성 (환경변수에서 인증정보 읽음)"""
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import Embeddings

    return Embeddings(
        model_id=EMBED_MODEL,
        credentials=Credentials(
            url=os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            api_key=os.environ["WATSONX_API_KEY"],
        ),
        project_id=os.environ["WATSONX_PROJECT_ID"],
    )


def main(chunks_path: str) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    chunks = json.loads(open(chunks_path, encoding="utf-8").read())
    print(f"청크 {len(chunks)}개 로드")

    embedder = get_embedder()
    client = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),  # 로컬이면 None이어도 됨
    )

    # ---- 컬렉션 준비 (없으면 생성) --------------------------------------
    # recreate가 아니라 "없을 때만 생성" — 재실행 시 기존 데이터를 날리지 않기 위함
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"컬렉션 '{COLLECTION}' 생성 (dim={EMBED_DIM})")

    # ---- 배치 단위로 임베딩 → 업로드 ------------------------------------
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]

        # ingest.py가 미리 계산해둔 embedding_text 사용 (없으면 즉석 생성)
        texts = [c.get("embedding_text") or f"{c['title']}\n{c['content']}" for c in batch]
        vectors = embedder.embed_documents(texts=texts)

        points = [
            PointStruct(
                id=c.get("chunk_id") or str(uuid.uuid4()),
                vector=v,
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
        print(f"  업로드 {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print("✅ 인덱싱 완료 — 이제 rag_query.py로 검색해볼 수 있어요")


if __name__ == "__main__":
    main(sys.argv[1])
