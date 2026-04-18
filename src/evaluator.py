# src/evaluator.py
"""
공정거래위원회 공모전 Track 2 - 평가 모듈
평가지표: Recall@5 (35%), MRR (15%), BERTScore (30%), F1 (25%)
"""

import re
from collections import Counter
from typing import List, Dict, Tuple


# ══════════════════════════════════════════════════════════════
# 1. 감점 방어 검증
# ══════════════════════════════════════════════════════════════


def validate_retrieval(retrieved_ids: List[str], valid_chunk_ids: set) -> List[str]:
    """
    감점 항목을 사전에 검증하고 오류 메시지 리스트를 반환합니다.
    오류가 없으면 빈 리스트를 반환합니다.

    감점 규칙:
        - 5개 미만/초과 반환       → 해당 문항 Retrieval 0점
        - 중복 chunk_id            → 해당 문항 Retrieval 0점
        - 존재하지 않는 chunk_id   → 해당 문항 전체 0점
    """
    errors = []

    if len(retrieved_ids) != 5:
        errors.append(f"5개가 아닌 {len(retrieved_ids)}개 반환 → Retrieval 0점")

    if len(retrieved_ids) != len(set(retrieved_ids)):
        dupes = [cid for cid, cnt in Counter(retrieved_ids).items() if cnt > 1]
        errors.append(f"중복 chunk_id {dupes} → Retrieval 0점")

    invalid = [cid for cid in retrieved_ids if cid not in valid_chunk_ids]
    if invalid:
        errors.append(f"존재하지 않는 chunk_id {invalid} → 문항 전체 0점")

    return errors


# ══════════════════════════════════════════════════════════════
# 2. Retrieval 평가: Recall@5, MRR
# ══════════════════════════════════════════════════════════════


def _recall_at_5(retrieved_ids: List[str], gt_id: str) -> float:
    """정답 chunk_id가 상위 5개 안에 있으면 1.0, 없으면 0.0"""
    return 1.0 if gt_id in retrieved_ids[:5] else 0.0


def _reciprocal_rank(retrieved_ids: List[str], gt_id: str) -> float:
    """정답 chunk_id의 순위에 따라 1/rank 반환. 없으면 0.0"""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid == gt_id:
            return 1.0 / rank
    return 0.0


# ══════════════════════════════════════════════════════════════
# 3. Generation 평가: Token F1
# ══════════════════════════════════════════════════════════════


def _tokenize(text: str) -> List[str]:
    """특수문자 제거 후 공백 단위 토크나이징 (한국어/영어 공용)"""
    return re.sub(r"[^\w\s]", " ", text).split()


def _token_f1(prediction: str, ground_truth: str) -> float:
    """토큰 수준 F1 점수 계산"""
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


# ══════════════════════════════════════════════════════════════
# 4. Generation 평가: BERTScore (배치 처리)
# ══════════════════════════════════════════════════════════════


def _bertscore_batch(
    predictions: List[str], ground_truths: List[str], lang: str = "ko"
) -> List[float]:
    """
    BERTScore를 배치로 한 번에 계산합니다. (개별 호출보다 수십 배 빠름)
    미설치 시 모두 0.0 반환.
    """
    try:
        from bert_score import score as bert_score_fn

        _, _, F = bert_score_fn(predictions, ground_truths, lang=lang, verbose=False)
        return F.tolist()
    except ImportError:
        print("[경고] bert-score 미설치. pip install bert-score 실행 후 사용하세요.")
        return [0.0] * len(predictions)


# ══════════════════════════════════════════════════════════════
# 5. 통합 배치 평가 (메인 인터페이스)
# ══════════════════════════════════════════════════════════════


def evaluate(
    retrieved_list: List[List[str]],
    gt_chunk_ids: List[str],
    predictions: List[str],
    gt_answers: List[str],
    valid_chunk_ids: set = None,
    use_bert_score: bool = True,
) -> Dict:
    """
    전체 데이터셋에 대한 배치 평가를 수행하고 최종 점수를 반환합니다.

    Args:
        retrieved_list  : 각 질문별 반환된 chunk_id 리스트 (질문 수 × 5)
        gt_chunk_ids    : 각 질문별 정답 chunk_id
        predictions     : 각 질문별 생성된 답변
        gt_answers      : 각 질문별 정답 텍스트
        valid_chunk_ids : 존재하는 chunk_id 전체 집합 (감점 방어용, 선택)
        use_bert_score  : BERTScore 계산 여부 (False로 끄면 빠름)

    Returns:
        {
            "Recall@5"          : float,
            "MRR"               : float,
            "F1"                : float,
            "BERTScore"         : float or None,
            "final_score"       : float,
            "per_question"      : List[Dict],   # 질문별 상세 점수
            "validation_errors" : List[str],    # 감점 경고 목록
        }

    가중치: Recall@5 (35%) | MRR (15%) | BERTScore (30%) | F1 (25%)
    """
    n = len(gt_chunk_ids)
    assert (
        len(retrieved_list) == n == len(predictions) == len(gt_answers)
    ), "retrieved_list, gt_chunk_ids, predictions, gt_answers 길이가 모두 같아야 합니다."

    recall_list, mrr_list, f1_list = [], [], []
    validation_errors = []
    per_question = []

    # ── 질문별 점수 계산 ──
    for i, (retrieved, gt_id, pred, gt_ans) in enumerate(
        zip(retrieved_list, gt_chunk_ids, predictions, gt_answers)
    ):
        # 감점 방어
        if valid_chunk_ids:
            errors = validate_retrieval(retrieved, valid_chunk_ids)
            if errors:
                validation_errors.extend([f"[Q{i+1}] {e}" for e in errors])

        recall = _recall_at_5(retrieved, gt_id)
        mrr = _reciprocal_rank(retrieved, gt_id)
        f1 = _token_f1(pred, gt_ans)

        recall_list.append(recall)
        mrr_list.append(mrr)
        f1_list.append(f1)

        per_question.append(
            {
                "question_idx": i + 1,
                "Recall@5": recall,
                "MRR": mrr,
                "F1": f1,
                "BERTScore": None,  # 아래에서 배치로 채움
            }
        )

    # ── BERTScore 배치 계산 (한 번에 처리) ──
    bert_scores = (
        _bertscore_batch(predictions, gt_answers) if use_bert_score else [0.0] * n
    )
    for i, bs in enumerate(bert_scores):
        per_question[i]["BERTScore"] = bs

    # ── 평균 계산 ──
    avg_recall = sum(recall_list) / n
    avg_mrr = sum(mrr_list) / n
    avg_f1 = sum(f1_list) / n
    avg_bert = sum(bert_scores) / n if use_bert_score else None

    bert_val = avg_bert if avg_bert is not None else 0.0
    final_score = avg_recall * 0.35 + avg_mrr * 0.15 + bert_val * 0.30 + avg_f1 * 0.25

    return {
        "Recall@5": avg_recall,
        "MRR": avg_mrr,
        "F1": avg_f1,
        "BERTScore": avg_bert,
        "final_score": final_score,
        "per_question": per_question,
        "validation_errors": validation_errors,
    }


# ══════════════════════════════════════════════════════════════
# 사용 예시
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    retrieved_list = [
        ["chunk_042", "chunk_007", "chunk_103", "chunk_055", "chunk_011"],
        ["chunk_010", "chunk_022", "chunk_031", "chunk_044", "chunk_060"],
    ]
    gt_chunk_ids = ["chunk_042", "chunk_099"]
    predictions = [
        "삼성전자에 과징금 5억원이 부과되었습니다.",
        "해당 행위는 불공정거래행위로 판단되었습니다.",
    ]
    gt_answers = [
        "과징금 5억원을 부과하였다.",
        "불공정거래행위에 해당한다고 결정하였다.",
    ]
    valid_chunk_ids = {f"chunk_{str(i).zfill(3)}" for i in range(1, 500)}

    scores = evaluate(
        retrieved_list=retrieved_list,
        gt_chunk_ids=gt_chunk_ids,
        predictions=predictions,
        gt_answers=gt_answers,
        valid_chunk_ids=valid_chunk_ids,
        use_bert_score=False,
    )

    print("=" * 50)
    print("📊 평가 결과")
    print("=" * 50)
    print(f"  Recall@5   (35%): {scores['Recall@5']:.4f}")
    print(f"  MRR        (15%): {scores['MRR']:.4f}")
    print(f"  BERTScore  (30%): {scores['BERTScore']}")
    print(f"  F1         (25%): {scores['F1']:.4f}")
    print(f"  최종 점수       : {scores['final_score']:.4f}")

    if scores["validation_errors"]:
        print("\n⚠️  감점 경고:")
        for err in scores["validation_errors"]:
            print(f"  - {err}")

    print("\n📋 질문별 점수:")
    for q in scores["per_question"]:
        print(
            f"  Q{q['question_idx']}: Recall={q['Recall@5']:.2f} MRR={q['MRR']:.2f} F1={q['F1']:.2f} BERT={q['BERTScore']}"
        )
