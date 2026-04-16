# src/config.py
import os

# 현재 파일(config.py)의 위치를 기준으로 프로젝트의 최상위 폴더(BASE_DIR) 경로를 찾습니다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터를 저장할 폴더들의 경로를 미리 지정해 둡니다.
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")  # 주최 측이 준 JSON 파일들을 넣을 폴더

# [RAG 검색 세팅값]
TOP_K = 5  # 질문을 던졌을 때, AI에게 넘겨줄 '가장 비슷한 문서 조각'의 개수입니다. (대회 규정: 5개)
RRF_K = (
    60  # 하이브리드 검색(의미+키워드) 점수를 합산할 때 쓰는 보정값 (보통 60을 씁니다)
)

# [AI 모델 세팅값]
# 문장을 숫자로 바꿔주는(임베딩) 국산 오픈소스 모델의 이름입니다. (한국어 성능이 좋습니다)
EMBEDDING_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
