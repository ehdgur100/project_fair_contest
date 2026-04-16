# main.py
import time
from src import config
from src.data_pipeline import load_provided_chunks
from src.retriever import build_vector_db, hybrid_search
from src.generator import generate_answer
from src.evaluator import (
    calculate_retrieval_metrics,
    token_f1_score,
    calculate_bertscore,
)


def run_pipeline():
    # --- [준비 단계] 도서관 세팅 ---
    print("🚀 [1단계] 제공된 JSON 데이터 로딩 중...")
    doc_chunks = load_provided_chunks(config.RAW_DATA_DIR)

    print("\n🔍 [2단계] 임베딩 및 하이브리드 검색 엔진 구축 중...")
    embed_model, faiss_index, bm25, chunk_ids, _ = build_vector_db(
        doc_chunks, config.EMBEDDING_MODEL_NAME
    )

    # --- [실전 단계] 테스트 세팅 ---
    test_query = "건설기계 임대단가를 결정하여 경쟁을 제한한 사업자단체에 대한 제재 조치는 무엇인가요?"
    ground_truth_chunk_ids = [chunk_ids[0]]  # 임의로 정한 정답 ID
    ground_truth_answer = "시정명령을 부과하고, 구성사업자에게 통지하도록 하였습니다."  # 임의로 정한 정답 텍스트

    start_time = time.time()  # ⏱️ 30초 타이머 켜기!

    # 1. 5개의 문서 번호표(ID) 찾기
    retrieved_chunk_ids = hybrid_search(
        test_query,
        embed_model,
        faiss_index,
        bm25,
        chunk_ids,
        config.TOP_K,
        config.RRF_K,
    )

    # 2. 번호표에 해당하는 진짜 텍스트 내용 가져오기
    retrieved_texts = []
    for chunk in doc_chunks:
        if chunk["chunk_id"] in retrieved_chunk_ids:
            retrieved_texts.append(chunk["text"])

    # 3. AI에게 진짜 텍스트를 주고 정답 작성시키기
    final_answer = generate_answer(test_query, retrieved_texts)

    end_time = time.time()  # ⏱️ 타이머 끄기!
    elapsed_time = end_time - start_time

    # --- [채점 단계] 결과 출력 ---
    print("\n📊 [3단계] 평가 결과")
    print(f"   [최종 답변]: {final_answer}")
    print(f"   [시간] 응답 소요 시간: {elapsed_time:.2f}초")

    # 평가 함수들을 호출하여 점수 매기기
    recall, mrr = calculate_retrieval_metrics(
        retrieved_chunk_ids, ground_truth_chunk_ids
    )
    f1 = token_f1_score(final_answer, ground_truth_answer)
    bert_f1 = calculate_bertscore(final_answer, ground_truth_answer)

    print(f"   [검색] Recall@5: {recall:.4f} | MRR: {mrr:.4f}")
    print(f"   [생성] Token F1: {f1:.4f} | BERTScore F1: {bert_f1:.4f}")


if __name__ == "__main__":
    run_pipeline()
