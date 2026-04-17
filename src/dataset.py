"""
공정거래위원회 공모전 Track 2 - 정답 데이터 자동 생성
서비스 주제: "내 일상 속 불공정거래, AI가 찾아드립니다" — 소비자 피해 공정거래 사례 통합 탐색기

대상 위반유형 3가지:
  - 부당한공동행위(담합)
  - 부당표시광고
  - 전자상거래소비자보호법위반

타겟: 일반 소비자
"""

import os
import json
import time
from typing import List, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv  # pip install python-dotenv
from openai import OpenAI


# ==========================================
# 1. 서비스 대상 위반유형 정의
# ==========================================

# 이 서비스에서 다루는 위반유형 키워드
TARGET_VIOLATION_KEYWORDS = [
    "부당한공동행위",
    "담합",
    "부당표시",
    "부당광고",
    "표시광고",
    "전자상거래",
]

# 위반유형별 소비자 질문 스타일 및 예시
VIOLATION_QUESTION_STYLE = {
    "담합": {
        "설명": "가격 담합으로 소비자가 더 비싸게 물건을 산 상황",
        "질문예시": [
            "라면 회사들이 가격을 같은 날 똑같이 올렸는데 담합인가요?",
            "이 사건에서 소비자들은 얼마나 피해를 봤나요?",
            "담합으로 어떤 제재를 받았나요?",
        ],
    },
    "표시광고": {
        "설명": "허위·과장 광고로 소비자가 속아 구매한 상황",
        "질문예시": [
            "이 제품 광고가 거짓이었나요?",
            "과장 광고로 얼마나 벌금을 받았나요?",
            "나도 비슷한 피해를 봤는데 신고할 수 있나요?",
        ],
    },
    "전자상거래": {
        "설명": "온라인 쇼핑 중 환불 거부, 계약 불이행 등 피해를 입은 상황",
        "질문예시": [
            "온라인 쇼핑몰에서 환불을 안 해줬는데 이런 경우 제재를 받나요?",
            "어떤 위반 행위로 제재를 받았나요?",
            "소비자는 어떻게 피해를 구제받을 수 있나요?",
        ],
    },
}


# ==========================================
# 2. 메타데이터에서 위반유형 확인 및 필터링
# ==========================================


def get_violation_category(metadata_path: str) -> Optional[str]:
    """
    metadata.json을 읽어 이 서비스 대상 위반유형인지 확인하고
    해당 카테고리("담합" / "표시광고" / "전자상거래")를 반환합니다.
    대상이 아니면 None을 반환합니다.

    Args:
        metadata_path: metadata.json 파일 경로

    Returns:
        "담합" / "표시광고" / "전자상거래" / None
    """
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        violation_types = [
            p.get("세부위반유형", p.get("위반유형", ""))
            for p in meta.get("피심인정보", [])
        ]

        for vtype in violation_types:
            vtype_clean = vtype.replace(" ", "")
            if any(k in vtype_clean for k in ["부당한공동행위", "담합"]):
                return "담합"
            if any(k in vtype_clean for k in ["부당표시", "부당광고", "표시광고"]):
                return "표시광고"
            if "전자상거래" in vtype_clean:
                return "전자상거래"

        return None  # 대상 위반유형 아님 → 건너뜀

    except Exception:
        return None


# ==========================================
# 3. GPT로 소비자 중심 질문-정답 쌍 생성
# ==========================================


def generate_qa_from_chunk(
    chunk: Dict,
    violation_category: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    qa_per_chunk: int = 2,
) -> List[Dict]:
    """
    청크 하나에서 GPT로 소비자 중심 질문-정답 쌍을 생성합니다.

    Args:
        chunk: {"chunk_id": ..., "text": ..., "metadata": ...}
        violation_category: "담합" / "표시광고" / "전자상거래"
        client: OpenAI 클라이언트
        model: 사용할 OpenAI 모델
        qa_per_chunk: 청크당 생성할 QA 쌍 수

    Returns:
        [{"question": ..., "gt_chunk_id": ..., "gt_answer": ..., "violation_category": ...}, ...]
    """
    if not chunk["text"].strip() or len(chunk["text"]) < 100:
        return []

    prompt = f"""당신은 공정거래위원회 의결서를 일반 소비자가 이해할 수 있도록 도와주는 AI입니다.

아래는 공정거래위원회 의결서의 일부입니다.

[의결서 텍스트]
{chunk["text"]}

위 텍스트를 읽고, 일반 소비자가 궁금해할 만한 질문과 정답 쌍을 {qa_per_chunk}개 생성하세요.

[규칙]
1. 질문은 위 텍스트만으로 답할 수 있어야 합니다.
2. 질문은 법률 용어 없이 일반인이 이해할 수 있는 쉬운 말로 작성하세요.
3. 정답은 텍스트 내용을 바탕으로 1~3문장으로 간결하게 작성하세요.
4. 정답도 쉬운 말로 풀어서 작성하세요. 어려운 법률 용어는 괄호로 설명을 덧붙이세요.
5. 반드시 아래 JSON 형식으로만 출력하세요. 다른 텍스트는 절대 포함하지 마세요.

[출력 형식]
[
  {{
    "question": "질문 내용",
    "answer": "정답 내용"
  }}
]"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        qa_list = json.loads(raw)

        result = []
        for qa in qa_list:
            if qa.get("question") and qa.get("answer"):
                result.append(
                    {
                        "question": qa["question"],
                        "gt_chunk_id": chunk["chunk_id"],
                        "gt_answer": qa["answer"],
                        "violation_category": violation_category,
                        "chunk_text": chunk["text"],  # 검수용
                    }
                )
        return result

    except Exception as e:
        print(f"[오류] {chunk['chunk_id']}: {e}")
        return []


# ==========================================
# 4. 전체 데이터셋 생성
# ==========================================


def generate_dataset(
    data_folder: str,
    output_path: str,
    client: OpenAI,
    max_chunks_per_doc: int = 3,
    qa_per_chunk: int = 2,
    model: str = "gpt-4o-mini",
) -> List[Dict]:
    """
    폴더 안의 의결서 중 대상 위반유형(담합/표시광고/전자상거래)만 필터링하여
    소비자 중심 QA 데이터셋을 생성합니다.

    Args:
        data_folder: hybrid.json 및 metadata.json이 있는 폴더 경로
        output_path: 생성된 데이터셋 저장 경로 (.json)
        client: OpenAI 클라이언트
        max_chunks_per_doc: 의결서당 사용할 최대 청크 수 (비용 절감)
        qa_per_chunk: 청크당 생성할 QA 쌍 수
        model: 사용할 OpenAI 모델

    Returns:
        전체 QA 데이터셋 리스트
    """
    hybrid_files = sorted(
        [f for f in os.listdir(data_folder) if f.endswith("_hybrid.json")]
    )

    if not hybrid_files:
        raise FileNotFoundError(
            f"'{data_folder}'에서 hybrid.json 파일을 찾을 수 없습니다."
        )

    print(f"전체 의결서: {len(hybrid_files)}개\n")

    all_qa = []

    for filename in tqdm(hybrid_files, desc="데이터셋 생성 중"):
        hybrid_path = os.path.join(data_folder, filename)

        # 청크 로딩
        try:
            with open(hybrid_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = [
                {
                    "chunk_id": item["metadata"]["chunk_id"],
                    "text": item["page_content"],
                    "metadata": item["metadata"],
                }
                for item in data
            ]
        except Exception as e:
            print(f"\n[오류] {filename} 로딩 실패: {e}")
            continue

        # 텍스트가 충분한 청크만 선택
        valid_chunks = [c for c in chunks if len(c["text"]) > 100]
        selected_chunks = valid_chunks[:max_chunks_per_doc]

        # QA 생성
        for chunk in selected_chunks:
            qa_list = generate_qa_from_chunk(
                chunk=chunk,
                violation_category="일반",
                client=client,
                model=model,
                qa_per_chunk=qa_per_chunk,
            )
            all_qa.extend(qa_list)
            time.sleep(0.5)  # rate limit 방어

    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 생성 완료")
    print(f"   - 총 QA 쌍:  {len(all_qa):,}개")
    print(f"   - 저장 경로: {output_path}")

    return all_qa


# ==========================================
# 5. 메인 실행
# ==========================================

if __name__ == "__main__":
    load_dotenv()  # .env 파일 로드
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    DATA_FOLDER = "data/raw/"
    OUTPUT_PATH = "data/ground_truth.json"

    client = OpenAI(api_key=OPENAI_API_KEY)

    dataset = generate_dataset(
        data_folder=DATA_FOLDER,
        output_path=OUTPUT_PATH,
        client=client,
        max_chunks_per_doc=3,  # 의결서당 최대 3개 청크 (비용 조절 가능)
        qa_per_chunk=2,  # 청크당 QA 2쌍 생성
        model="gpt-4o-mini",
    )

    # 샘플 출력
    print("\n--- 생성된 QA 샘플 ---")
    for qa in dataset[:3]:
        print(f"\n[질문]     {qa['question']}")
        print(f"[정답]     {qa['gt_answer']}")
        print(f"[chunk_id] {qa['gt_chunk_id']}")
