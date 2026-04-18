"""
공정거래위원회 공모전 Track 2 - 이미지 처리 모듈
PDF에서 이미지를 추출하고 GPT-4o Vision으로 설명을 생성한 뒤
hybrid.json의 chunk_id와 매칭하여 청크 텍스트를 보강합니다.

설치 필요:
    pip install pypdfium2 pillow openai python-dotenv
"""

import os
import json
import base64
import re
import time
from io import BytesIO
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

import pypdfium2 as pdfium
from PIL import Image


# ══════════════════════════════════════════════════════════════
# 1. PDF 페이지를 이미지로 변환
# ══════════════════════════════════════════════════════════════


def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    """
    PDF의 각 페이지를 PIL Image 객체 리스트로 변환합니다.

    Args:
        pdf_path : PDF 파일 경로
        dpi      : 해상도 (높을수록 선명하지만 느림, 150이 속도/품질 균형)

    Returns:
        페이지별 PIL Image 리스트 (인덱스 0 = 1페이지)
    """
    pdf = pdfium.PdfDocument(pdf_path)
    scale = dpi / 72  # 72가 기본 DPI
    images = []
    for page in pdf:
        bitmap = page.render(scale=scale)
        images.append(bitmap.to_pil())
    print(f"총 {len(images)}페이지 변환 완료")
    return images


# ══════════════════════════════════════════════════════════════
# 2. 이미지에 텍스트가 있는지 확인 (처리 필요 여부 판별)
# ══════════════════════════════════════════════════════════════


def has_meaningful_image(page_image: Image.Image, text_threshold: float = 0.85) -> bool:
    """
    페이지가 텍스트만으로 구성됐는지, 의미 있는 이미지가 포함됐는지 판별합니다.
    픽셀 밝기 분포로 간단히 판별합니다. (완벽하지 않으나 비용 절감에 유용)

    Args:
        page_image      : PIL Image 객체
        text_threshold  : 흰색 픽셀 비율이 이 값 이상이면 텍스트 전용 페이지로 판단

    Returns:
        True = 이미지 처리 필요, False = 텍스트 전용 페이지
    """
    gray = page_image.convert("L")
    pixels = list(gray.getdata())
    white_ratio = sum(1 for p in pixels if p > 200) / len(pixels)
    return white_ratio < text_threshold


# ══════════════════════════════════════════════════════════════
# 3. GPT-4o Vision으로 이미지 설명 생성
# ══════════════════════════════════════════════════════════════


def image_to_base64(image: Image.Image) -> str:
    """PIL Image를 base64 문자열로 변환합니다."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def describe_page_image(
    page_image: Image.Image,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    """
    GPT-4o Vision으로 페이지 이미지를 분석하고 텍스트 설명을 생성합니다.

    Args:
        page_image : PIL Image 객체
        client     : OpenAI 클라이언트
        model      : Vision 지원 모델 (gpt-4o 또는 gpt-4o-mini)

    Returns:
        이미지 설명 텍스트 (처리 실패 시 None)
    """
    b64 = image_to_base64(page_image)

    prompt = """이 이미지는 공정거래위원회 의결서의 일부입니다.
이미지에 포함된 시각적 요소를 텍스트로 정확하게 설명해주세요.

다음 내용이 있다면 반드시 포함하세요:
- 흐름도/다이어그램: 단계와 순서, 각 항목명
- 캡처 이미지(카카오톡, 이메일, 문서 스크린샷 등): 내용 전문
- 그래프/차트: 수치와 항목
- 제품 사진: 제품명과 특징

비공개 처리된 부분은 "(비공개)"로 표기하세요.
표(테이블)는 설명하지 않아도 됩니다.
흐름도/캡처/그래프/사진 등 시각적 요소가 전혀 없는 경우에만 "이미지 없음"이라고 답하세요."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        result = response.choices[0].message.content.strip()

        # "이미지 없음" 포함 여부로 필터링 (정확한 문자열 매칭보다 넓게 체크)
        if "이미지 없음" in result:
            return None

        # 표 설명만 있는 경우 필터링
        # GPT가 표를 설명했더라도 흐름도/캡처/그래프/사진 언급이 없으면 제외
        visual_keywords = [
            "흐름도",
            "다이어그램",
            "캡처",
            "카카오톡",
            "이메일",
            "스크린샷",
            "그래프",
            "차트",
            "사진",
            "그림",
            "제품",
        ]
        if not any(kw in result for kw in visual_keywords):
            return None

        return result

    except Exception as e:
        print(f"[오류] Vision API 호출 실패: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# 4. 이미지 설명을 가장 적합한 청크에 매칭
# ══════════════════════════════════════════════════════════════


def _text_similarity(text_a: str, text_b: str) -> float:
    """두 텍스트의 단어 겹침 비율로 유사도를 계산합니다. (Jaccard 유사도)"""
    tokens_a = set(re.sub(r"[^\w\s]", " ", text_a).split())
    tokens_b = set(re.sub(r"[^\w\s]", " ", text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def match_image_to_chunk(
    image_description: str,
    chunks: List[Dict],
    page_text: str = "",
) -> Optional[str]:
    """
    이미지 설명을 가장 관련성 높은 청크의 chunk_id에 매칭합니다.

    Args:
        image_description : GPT-4o Vision이 생성한 설명 텍스트
        chunks            : hybrid.json에서 로드한 전체 청크 리스트
        page_text         : 해당 PDF 페이지의 텍스트 (범위 좁히기용)

    Returns:
        가장 적합한 chunk_id (매칭 실패 시 None)
    """
    # 페이지 텍스트와 유사한 청크로 후보 범위 좁히기
    if page_text.strip():
        candidates = sorted(
            chunks,
            key=lambda c: _text_similarity(page_text, c["text"]),
            reverse=True,
        )[:5]
    else:
        candidates = chunks

    # 이미지 설명과 가장 유사한 청크 선택
    best_chunk_id = None
    best_score = 0.0

    for chunk in candidates:
        score = _text_similarity(image_description, chunk["text"])
        if score > best_score:
            best_score = score
            best_chunk_id = chunk["chunk_id"]

    return best_chunk_id if best_score > 0.05 else None


# ══════════════════════════════════════════════════════════════
# 5. 청크에 이미지 설명 추가 (보강)
# ══════════════════════════════════════════════════════════════


def enrich_chunks_with_images(
    pdf_path: str,
    hybrid_json_path: str,
    output_path: str,
    client: OpenAI,
    vision_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """
    PDF의 이미지를 분석하여 hybrid.json 청크를 보강합니다.

    처리 흐름:
        PDF → 페이지 이미지 → 의미있는 이미지 필터링
        → GPT-4o Vision 설명 생성 → 청크 매칭 → 청크 텍스트에 설명 추가
        → 보강된 JSON 저장

    Args:
        pdf_path         : PDF 파일 경로
        hybrid_json_path : 원본 hybrid.json 경로
        output_path      : 보강된 결과 저장 경로
        client           : OpenAI 클라이언트
        vision_model     : Vision 모델명

    Returns:
        보강된 청크 리스트
    """
    # 1. 청크 로딩
    with open(hybrid_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    chunks = [
        {
            "chunk_id": item["metadata"]["chunk_id"],
            "text": item["page_content"],
            "metadata": item["metadata"],
        }
        for item in raw_data
    ]
    print(f"청크 {len(chunks)}개 로딩 완료")

    # 2. PDF → 페이지 이미지 변환
    page_images = pdf_to_images(pdf_path)

    # 3. 페이지별 처리
    enriched_count = 0
    chunk_map = {c["chunk_id"]: c for c in chunks}

    for page_num, page_image in enumerate(page_images, start=1):
        print(f"  [페이지 {page_num:2d}] Vision 분석 중...")

        # 4. GPT-4o Vision으로 설명 생성
        description = describe_page_image(page_image, client, vision_model)
        if not description:
            print(f"  [페이지 {page_num:2d}] 유의미한 이미지 없음 → 스킵")
            continue

        print(f"  [페이지 {page_num:2d}] 설명 생성 완료: {description[:60]}...")

        # 5. 가장 적합한 청크에 매칭
        matched_id = match_image_to_chunk(description, chunks)
        if not matched_id:
            print(f"  [페이지 {page_num:2d}] 매칭 실패 → 스킵")
            continue

        # 6. 청크 텍스트에 이미지 설명 추가
        chunk_map[matched_id][
            "text"
        ] += f"\n\n[이미지 설명 - 페이지 {page_num}]\n{description}"
        enriched_count += 1
        print(f"  [페이지 {page_num:2d}] → {matched_id} 에 추가 완료")

        # rate limit 방어: 페이지마다 3초 대기
        time.sleep(3)

    # 7. 보강된 결과 저장
    enriched_chunks = list(chunk_map.values())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 완료: {enriched_count}개 청크 보강됨 → {output_path}")
    return enriched_chunks


# ══════════════════════════════════════════════════════════════
# 6. 전체 폴더 일괄 처리
# ══════════════════════════════════════════════════════════════


def process_all_documents(
    data_folder: str,
    output_folder: str,
    client: OpenAI,
    vision_model: str = "gpt-4o-mini",
) -> None:
    """
    data_folder 안의 모든 의결서 세트(PDF + hybrid.json)를 일괄 처리합니다.

    Args:
        data_folder   : PDF와 hybrid.json이 있는 폴더
        output_folder : 보강된 hybrid.json을 저장할 폴더
        client        : OpenAI 클라이언트
        vision_model  : Vision 모델명
    """
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    print(f"총 {len(pdf_files)}개 PDF 발견\n")

    for pdf_filename in pdf_files:
        base_name = pdf_filename.replace(".pdf", "")
        pdf_path = os.path.join(data_folder, pdf_filename)
        hybrid_path = os.path.join(data_folder, f"{base_name}_hybrid.json")
        output_path = os.path.join(output_folder, f"{base_name}_hybrid_enriched.json")

        if not os.path.exists(hybrid_path):
            print(f"[스킵] {base_name}: hybrid.json 없음")
            continue

        print(f"\n{'='*60}")
        print(f"처리 중: {base_name}")
        print(f"{'='*60}")

        try:
            enrich_chunks_with_images(
                pdf_path=pdf_path,
                hybrid_json_path=hybrid_path,
                output_path=output_path,
                client=client,
                vision_model=vision_model,
            )
        except Exception as e:
            print(f"[오류] {base_name}: {e}")
            continue


# ══════════════════════════════════════════════════════════════
# 7. 메인 실행
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # 단일 파일 테스트
    enrich_chunks_with_images(
        pdf_path="data/raw/(주)놀유니버스 및 (주)야놀자의 거래상지위남용행위 등에 대한 건.pdf",
        hybrid_json_path="data/raw/(주)놀유니버스 및 (주)야놀자의 거래상지위남용행위 등에 대한 건_hybrid.json",
        output_path="data/output/(주)놀유니버스 및 (주)야놀자의 거래상지위남용행위 등에 대한 건_hybrid_enriched.json",
        client=client,
        vision_model="gpt-4o-mini",
    )

    # 전체 폴더 일괄 처리 시
    # process_all_documents(
    #     data_folder   = "data/raw/",
    #     output_folder = "data/output/",
    #     client        = client,
    # )
