# main.py
import json
import time

from src import config
from src.data_pipeline import load_provided_chunks
from src.retriever import build_vector_db, hybrid_search
from src.generator import generate_answer
from src.evaluator import evaluate  # [변경] 단일 함수 import로 통합


def run_pipeline():
    # ── [준비 단계] ──────────────────────────────────────────
    print("🚀 [1단계] 제공된 JSON 데이터 로딩 중...")
    doc_chunks = load_provided_chunks(config.RAW_DATA_DIR)

    # [변경] 감점 방어용 valid_chunk_ids 집합 생성
    valid_chunk_ids = {chunk["chunk_id"] for chunk in doc_chunks}

    print("\n🔍 [2단계] 임베딩 및 하이브리드 검색 엔진 구축 중...")
    embed_model, faiss_index, bm25, chunk_ids, _ = build_vector_db(
        doc_chunks, config.EMBEDDING_MODEL_NAME
    )

    # ── [변경] ground_truth.json에서 평가 데이터 로드 ────────
    print("\n📂 [3단계] 평가 데이터(ground_truth.json) 로딩 중...")
    with open(config.GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    print(f"   총 {len(ground_truth_data)}개 질문 로드 완료")

    # ── [변경] 전체 질문 루프 ─────────────────────────────────
    print("\n⚙️  [4단계] 전체 질문 평가 중...")

    retrieved_list = []
    predictions = []
    gt_chunk_ids = []
    gt_answers = []

    for item in ground_truth_data:
        query = item["question"]
        gt_id = item["gt_chunk_id"]
        gt_ans = item["gt_answer"]

        start_time = time.time()

        # 1. Retrieval
        retrieved_ids = hybrid_search(
            query,
            embed_model,
            faiss_index,
            bm25,
            chunk_ids,
            config.TOP_K,
            config.RRF_K,
        )

        # 2. Generation — chunk_id에 해당하는 텍스트 꺼내서 전달
        retrieved_texts = [
            chunk["text"] for chunk in doc_chunks if chunk["chunk_id"] in retrieved_ids
        ]
        answer = generate_answer(query, retrieved_texts)

        elapsed = time.time() - start_time

        # [변경] 30초 초과 경고
        if elapsed > 30:
            print(f"  ⚠️  응답 시간 초과 ({elapsed:.1f}초) → 문항 전체 0점 위험")

        retrieved_list.append(retrieved_ids)
        predictions.append(answer)
        gt_chunk_ids.append(gt_id)
        gt_answers.append(gt_ans)

    # ── [변경] evaluator.evaluate()로 배치 평가 ──────────────
    print("\n📊 [5단계] 평가 결과")
    print("=" * 50)

    scores = evaluate(
        retrieved_list=retrieved_list,
        gt_chunk_ids=gt_chunk_ids,
        predictions=predictions,
        gt_answers=gt_answers,
        valid_chunk_ids=valid_chunk_ids,
        use_bert_score=True,
    )

    print(f"  Recall@5   (35%): {scores['Recall@5']:.4f}")
    print(f"  MRR        (15%): {scores['MRR']:.4f}")
    print(
        f"  BERTScore  (30%): {scores['BERTScore']:.4f}"
        if scores["BERTScore"]
        else "  BERTScore  (30%): None"
    )
    print(f"  F1         (25%): {scores['F1']:.4f}")
    print("-" * 50)
    print(f"  최종 점수       : {scores['final_score']:.4f}")

    if scores["validation_errors"]:
        print("\n⚠️  감점 경고:")
        for err in scores["validation_errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    run_pipeline()
