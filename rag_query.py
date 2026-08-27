"""
⑥ RAG 검색 + 답변 생성 — "꺼내 쓰기" 단계
====================================================

사용법:
    python rag_query.py "로그인 기능은 누가 개발했어?"

흐름 (이게 RAG의 전부다):
    1. 질문을 같은 임베딩 모델로 벡터화
       (★반드시 인덱싱 때와 같은 모델 — 다른 모델이면 벡터 공간이 달라 검색 불가)
    2. Qdrant에서 유사도 상위 k개 청크 검색
    3. 검색된 청크를 근거로 붙여서 watsonx LLM에게 답변 생성 요청
"""

from __future__ import annotations

import os
import sys

from embed_and_upload import COLLECTION, get_embedder

TOP_K = 5  # 검색해서 LLM에게 보여줄 청크 수 (많을수록 근거↑ 비용↑)


def search(question: str) -> list:
    """질문 → 임베딩 → Qdrant 유사도 검색"""
    from qdrant_client import QdrantClient

    embedder = get_embedder()
    # embed_query: 검색 질의용 임베딩 (문서용 embed_documents와 짝)
    q_vector = embedder.embed_query(text=question)

    client = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
    hits = client.query_points(
        collection_name=COLLECTION,
        query=q_vector,
        limit=TOP_K,
        with_payload=True,
    ).points
    return hits


def answer(question: str, hits: list) -> str:
    """검색된 청크를 근거로 watsonx LLM이 답변 생성"""
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    # 검색 결과를 "번호 붙은 근거 목록" 텍스트로 조립
    # → LLM이 어떤 근거로 답했는지 [1], [2]로 인용하게 만든다 (환각 방지)
    context_parts = []
    for i, h in enumerate(hits, 1):
        p = h.payload
        context_parts.append(
            f"[{i}] ({p['source_type']}) {p['title']}\n{p['content'][:500]}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""당신은 팀 프로젝트 기여도 분석 도우미입니다.
아래 [근거 자료]만 사용해서 질문에 답하세요.
근거에 없는 내용은 지어내지 말고 "자료에 없음"이라고 답하세요.
답변에 사용한 근거 번호를 [1]처럼 표시하세요.

[근거 자료]
{context}

[질문]
{question}

[답변]"""

    model = ModelInference(
        model_id="ibm/granite-3-8b-instruct",  # 팀에서 쓰기로 한 모델로 교체 가능
        credentials=Credentials(
            url=os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            api_key=os.environ["WATSONX_API_KEY"],
        ),
        project_id=os.environ["WATSONX_PROJECT_ID"],
    )
    return model.generate_text(prompt=prompt, params={"max_new_tokens": 500})


if __name__ == "__main__":
    question = sys.argv[1]

    hits = search(question)
    print("=== 검색된 근거 ===")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h.score:.3f} ({h.payload['source_type']}) {h.payload['title']}")

    print("\n=== 답변 ===")
    print(answer(question, hits))
