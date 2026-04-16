# src/evaluator.py
import time
from typing import List
from collections import Counter
from bert_score import score as bert_score


def calculate_retrieval_metrics(retrieved_ids: List[str], ground_truth_ids: List[str]):
    """검색 능력을 평가합니다. (Recall@5: 정답 문서가 5개 안에 있는가?)"""
    hits = len(
        set(retrieved_ids) & set(ground_truth_ids)
    )  # 교집합을 찾아 맞춘 개수 계산
    recall_at_5 = hits / len(ground_truth_ids) if ground_truth_ids else 0.0

    mrr = 0.0  # 정답 문서가 1등에 가까울수록 점수를 높게 줍니다.
    for i, rid in enumerate(retrieved_ids):
        if rid in ground_truth_ids:
            mrr = 1.0 / (i + 1)
            break
    return recall_at_5, mrr


def token_f1_score(prediction: str, ground_truth: str):
    """답변의 '글자(토큰)' 일치도를 평가합니다. (조사 하나만 달라도 깎이는 무서운 지표)"""
    pred_tokens = prediction.split()  # AI의 답변을 띄어쓰기 단위로 쪼갬
    truth_tokens = ground_truth.split()  # 내 정답지를 띄어쓰기 단위로 쪼갬

    common = Counter(pred_tokens) & Counter(truth_tokens)  # 겹치는 단어 찾기
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)  # F1 수학 공식 적용


def calculate_bertscore(prediction: str, ground_truth: str):
    """답변의 '의미(문맥)' 일치도를 평가합니다. (말만 통하면 점수를 잘 줍니다)"""
    # AI 언어모델(BERT)을 활용해 두 문장의 뜻이 비슷한지 수학적으로 검사합니다.
    P, R, F1_bert = bert_score([prediction], [ground_truth], lang="ko", verbose=False)
    return F1_bert.mean().item()
