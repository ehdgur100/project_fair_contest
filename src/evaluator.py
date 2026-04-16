# src/evaluator.py

import time
from typing import List
from collections import Counter
from bert_score import score as bert_score


def calculate_retrieval_metrics(retrieved_ids: List[str], ground_truth_ids: List[str]):
    """Recall@5 와 MRR 계산"""
    hits = len(set(retrieved_ids) & set(ground_truth_ids))
    recall_at_5 = hits / len(ground_truth_ids) if ground_truth_ids else 0.0

    mrr = 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in ground_truth_ids:
            mrr = 1.0 / (i + 1)
            break
    return recall_at_5, mrr


def token_f1_score(prediction: str, ground_truth: str):
    """토큰 수준 F1 스코어 계산"""
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def calculate_bertscore(prediction: str, ground_truth: str):
    """
    BERTScore (F1) 계산
    에러 방지를 위해 model_type을 직접 지정하지 않고 언어 설정(ko)만 사용합니다.
    """
    # model_type 설정을 제거하여 기본 권장 모델을 사용하도록 수정
    P, R, F1_bert = bert_score([prediction], [ground_truth], lang="ko", verbose=False)
    return F1_bert.mean().item()
