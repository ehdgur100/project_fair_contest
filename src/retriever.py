# src/retriever.py
import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


def build_vector_db(doc_chunks: List[Dict], model_name: str):
    """문서를 검색할 수 있도록 수학적 공간(DB)을 구축하는 함수입니다."""

    # 1. 한국어 문장을 숫자로 바꿔줄 AI 모델을 불러옵니다.
    embed_model = SentenceTransformer(model_name, device="cpu")

    texts = [doc["text"] for doc in doc_chunks]
    chunk_ids = [doc["chunk_id"] for doc in doc_chunks]

    # 2. FAISS (의미 기반 검색) 구축
    # 문장들을 AI를 통해 긴 숫자 배열(벡터)로 바꾸고, FAISS라는 고속 검색기에 넣습니다.
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 3. BM25 (키워드 기반 검색) 구축
    # 문장을 띄어쓰기 단위로 쪼개서, 특정 단어가 얼마나 자주 나오는지 통계를 냅니다.
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    return embed_model, index, bm25, chunk_ids, texts


def hybrid_search(
    query: str, embed_model, faiss_index, bm25, chunk_ids, top_k: int, rrf_k: int
) -> List[str]:
    """두 가지 검색 결과를 짬짜면처럼 스까서(RRF) 가장 좋은 5개를 뽑는 함수입니다."""

    # 1. FAISS 검색: 질문과 의미가 비슷한 순서대로 등수를 매깁니다.
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    dense_distances, dense_indices = faiss_index.search(query_emb, top_k * 10)

    # 2. BM25 검색: 질문 속 단어가 똑같이 들어간 순서대로 등수를 매깁니다.
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][: top_k * 10]

    # 3. RRF (Reciprocal Rank Fusion) 방식 적용
    # 두 검색 결과의 등수를 합산하여 최종 점수를 계산합니다. (1등은 점수를 많이 줌)
    rrf_scores = {cid: 0.0 for cid in chunk_ids}

    for rank, idx in enumerate(dense_indices[0]):
        cid = chunk_ids[idx]
        rrf_scores[cid] += 1.0 / (rrf_k + rank + 1)  # FAISS 점수 더하기

    for rank, idx in enumerate(bm25_indices):
        cid = chunk_ids[idx]
        rrf_scores[cid] += 1.0 / (rrf_k + rank + 1)  # BM25 점수 더하기

    # 4. 최종 점수가 가장 높은 순서대로 정렬하여 상위 5개의 ID만 잘라서 반환합니다.
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, score in sorted_chunks[:top_k]]
