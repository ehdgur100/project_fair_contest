import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

# 🔥 1. 형태소 분석기 초기화
kiwi = Kiwi()


def tokenize_korean(text: str) -> List[str]:
    """한국어 문장을 형태소(명사, 동사 등) 단위로 정밀하게 쪼갭니다."""
    return [token.form for token in kiwi.tokenize(text)]


def build_vector_db(doc_chunks: List[Dict], model_name: str):
    """문서를 검색할 수 있도록 수학적 공간(DB)을 구축하는 함수입니다."""

    embed_model = SentenceTransformer(model_name, device="cpu")

    # 🚨 팀장님의 _hybrid.json 실제 구조에 완벽하게 맞춤 (page_content, metadata)
    texts = [doc["page_content"] for doc in doc_chunks]
    chunk_ids = [doc["metadata"]["chunk_id"] for doc in doc_chunks]

    # 2. FAISS 업그레이드 (Cosine Similarity 적용)
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)  # 내적(IP) 전 벡터 길이 정규화 필수
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 코사인 유사도(IP) 엔진 장착
    index.add(embeddings)

    # 3. BM25 업그레이드 (Kiwi 형태소 분석기 적용)
    tokenized_texts = [tokenize_korean(text) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    return embed_model, index, bm25, chunk_ids, texts


def hybrid_search(
    query: str,
    embed_model,
    faiss_index,
    bm25,
    chunk_ids,
    top_k: int,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.5,
) -> List[str]:
    """오류 방어 로직이 추가된 가중치 기반 하이브리드 검색 함수"""

    # 🚨 FAISS Out of Bounds 방어: DB에 있는 문서 수보다 많이 검색하지 않도록 제한
    search_k = min(top_k * 10, len(chunk_ids))

    # 1. FAISS 의미 검색
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    dense_distances, dense_indices = faiss_index.search(query_emb, search_k)

    # 2. BM25 키워드 검색 (질문도 형태소로 쪼개기)
    tokenized_query = tokenize_korean(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:search_k]

    # 3. Weighted RRF 계산 (BM25 키워드 점수에 가중치 1.5배)
    rrf_scores = {cid: 0.0 for cid in chunk_ids}

    # 의미 검색 점수 합산
    for rank, idx in enumerate(dense_indices[0]):
        if idx == -1:
            continue  # 🚨 혹시 모를 FAISS의 -1 반환값 무시 (에러 방어)
        cid = chunk_ids[idx]
        rrf_scores[cid] += dense_weight * (1.0 / (rrf_k + rank + 1))

    # 키워드 검색 점수 합산
    for rank, idx in enumerate(bm25_indices):
        cid = chunk_ids[idx]
        rrf_scores[cid] += sparse_weight * (1.0 / (rrf_k + rank + 1))

    # 4. 최종 정렬 및 상위 반환
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, score in sorted_chunks[:top_k]]
