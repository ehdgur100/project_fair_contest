# src/data_pipeline.py
import os
import glob
import json
from typing import List, Dict


def load_provided_chunks(directory_path: str) -> List[Dict]:
    """주최 측에서 제공한 _hybrid.json 파일들을 읽어옵니다."""
    # 하위 폴더까지 포함하여 _hybrid.json 파일들을 찾습니다.
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
    all_doc_chunks = []

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # JSON 리스트 내의 각 청크 데이터를 순회합니다.
                for item in data:
                    text = item.get("page_content", "")
                    # metadata 내부에 chunk_id가 위치한 구조를 반영합니다.
                    chunk_id = item.get("metadata", {}).get("chunk_id", "")

                    if text and chunk_id:
                        all_doc_chunks.append({"chunk_id": chunk_id, "text": text})
        except Exception as e:
            print(f"[오류] {os.path.basename(json_path)} 읽기 실패: {e}")
            continue

    return all_doc_chunks
