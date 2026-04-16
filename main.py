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
    print("🚀 [1단계] 제공된 JSON 데이터 로딩 중...")
    doc_chunks = load_provided_chunks(config.RAW_DATA_DIR)
    print(f"   ✓ 로드된 총 청크 개수: {len(doc_chunks)}개")

    if not doc_chunks:
        print("🚨 데이터가 없습니다. 프로그램을 종료합니다.")
        return

    print(
        "\n🔍 [2단계] 임베딩 및 하이브리드 검색 엔진 구축 중... (약간의 시간이 소요될 수 있습니다)"
    )
    embed_model, faiss_index, bm25, chunk_ids, _ = build_vector_db(
        doc_chunks, config.EMBEDDING_MODEL_NAME
    )

    # ------------------ 평가 시작 ------------------
    # 업로드해주신 '건설기계개별연명사업자협의회' 관련 질문
    test_query = "건설기계 임대단가를 결정하여 경쟁을 제한한 사업자단체에 대한 제재 조치는 무엇인가요?"
    ground_truth_chunk_ids = [chunk_ids[0]]  # 임의의 정답 설정
    ground_truth_answer = "시정명령을 부과하고, 구성사업자에게 통지하도록 하였습니다."

    print(f"\n💡 질문: {test_query}")

    start_time = time.time()

    # 검색 (정확히 TOP_K개 반환)
    retrieved_chunk_ids = hybrid_search(
        test_query,
        embed_model,
        faiss_index,
        bm25,
        chunk_ids,
        config.TOP_K,
        config.RRF_K,
    )
    retrieved_texts = []
    for chunk in doc_chunks:
        if chunk["chunk_id"] in retrieved_chunk_ids:
            retrieved_texts.append(chunk["text"])

    # 생성 (Mock)
    final_answer = generate_answer(test_query, retrieved_texts)

    print("\n[AI에게 전달된 5개의 문서 내용 확인]")
    for i, text in enumerate(retrieved_texts):
        print(f"--- 청크 {i+1} ---")
        print(text[:100] + " ... (중략)")

    end_time = time.time()
    elapsed_time = end_time - start_time
    # -----------------------------------------------

    print("\n📊 [3단계] 평가 결과")
    print(f"   [최종 답변]: {final_answer}")
    print(f"   [검색된 5개 청크 ID]: {retrieved_chunk_ids}")
    print(f"   [시간] 응답 소요 시간: {elapsed_time:.2f}초", end="")
    if elapsed_time > 30:
        print(" 🚨 (감점 위험: 30초 초과!)")
    else:
        print(" ✅ (통과)")

    # 지표 계산
    recall, mrr = calculate_retrieval_metrics(
        retrieved_chunk_ids, ground_truth_chunk_ids
    )
    f1 = token_f1_score(final_answer, ground_truth_answer)
    bert_f1 = calculate_bertscore(final_answer, ground_truth_answer)

    print(f"   [검색] Recall@5: {recall:.4f} | MRR: {mrr:.4f}")
    print(f"   [생성] Token F1: {f1:.4f} | BERTScore F1: {bert_f1:.4f}")


if __name__ == "__main__":
    run_pipeline()
