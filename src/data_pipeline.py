# src/data_pipeline.py
import os
import glob
import json
from typing import List, Dict


def load_provided_chunks(directory_path: str) -> List[Dict]:
    """주최 측에서 제공한 _hybrid.json 파일들을 읽어옵니다."""

    # 1. glob.glob를 써서 지정된 폴더 안의 모든 '*_hybrid.json' 파일 경로를 싹 찾습니다.
    json_files = glob.glob(
        os.path.join(directory_path, "**", "*_hybrid.json"), recursive=True
    )

    if not json_files:
        print(
            f"경고: '{directory_path}' 폴더에서 _hybrid.json 파일을 찾을 수 없습니다!"
        )
        return []

    print(
        f"총 {len(json_files)}개의 _hybrid.json 파일을 발견했습니다. 로딩을 시작합니다..."
    )
    all_doc_chunks = []  # 정리된 데이터를 담을 큰 바구니

    # 2. 찾은 파일들을 하나씩 열어봅니다.
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # JSON 파일을 파이썬 딕셔너리로 변환

                # 3. 파일 안의 데이터 조각(Chunk)들을 하나씩 꺼내서 정리합니다.
                for item in data:
                    text = item.get("page_content", "")  # 실제 문서 본문 텍스트
                    chunk_id = item.get("metadata", {}).get(
                        "chunk_id", ""
                    )  # 문서의 고유 ID표

                    # 텍스트와 ID가 둘 다 무사히 존재한다면 바구니에 담습니다.
                    if text and chunk_id:
                        all_doc_chunks.append({"chunk_id": chunk_id, "text": text})
        except Exception as e:
            print(f"[오류] {os.path.basename(json_path)} 읽기 실패: {e}")
            continue

    return all_doc_chunks  # 꽉 찬 바구니를 반환합니다.
