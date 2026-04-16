# src/retriever.py
import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


def build_vector_db(doc_chunks: List[Dict], model_name: str):
    """Dense(FAISS)와 Sparse(BM25) 인덱스를 구축합니다."""
    # 노트북 환경에 맞춰 CPU 구동 명시
    embed_model = SentenceTransformer(model_name, device="cpu")

    texts = [doc["text"] for doc in doc_chunks]
    chunk_ids = [doc["chunk_id"] for doc in doc_chunks]

    # FAISS 구축
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # BM25 구축
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    return embed_model, index, bm25, chunk_ids, texts


def hybrid_search(
    query: str, embed_model, faiss_index, bm25, chunk_ids, top_k: int, rrf_k: int
) -> List[str]:
    """RRF 기반 하이브리드 검색 수행 (정확히 top_k개 반환)"""
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    dense_distances, dense_indices = faiss_index.search(query_emb, top_k * 10)

    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][: top_k * 10]

    rrf_scores = {cid: 0.0 for cid in chunk_ids}

    for rank, idx in enumerate(dense_indices[0]):
        cid = chunk_ids[idx]
        rrf_scores[cid] += 1.0 / (rrf_k + rank + 1)

    for rank, idx in enumerate(bm25_indices):
        cid = chunk_ids[idx]
        rrf_scores[cid] += 1.0 / (rrf_k + rank + 1)

    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, score in sorted_chunks[:top_k]]
