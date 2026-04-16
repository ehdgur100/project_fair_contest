# src/generator.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 내 컴퓨터의 .env 파일을 읽어옵니다.
load_dotenv()

# 2. 환경 변수에서 API 키를 안전하게 꺼내옵니다.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# src/generator.py
def generate_answer(query: str, retrieved_chunks: list) -> str:
    """검색된 5개의 청크를 바탕으로 GPT-4o-mini가 실제 답변을 생성합니다."""

    # 만약 검색된 청크가 없다면 바로 예외 처리
    if not retrieved_chunks:
        return "검색된 관련 문서가 없어 답변할 수 없습니다."

    # 1. 벡터 DB에서 찾아온 텍스트 조각들을 하나의 긴 문장으로 합치기
    context = "\n\n---\n\n".join(retrieved_chunks)

    # 2. AI에게 부여할 역할과 규칙 (프롬프트 엔지니어링의 핵심!)
    system_prompt = """
    당신은 대한민국 공정거래위원회 의결서를 분석하는 '법률 AI 어시스턴트'입니다.
    반드시 아래 제공된 [문서 내용]만을 근거로 하여 질문에 답변하세요.
    문서 내용에 없는 정보는 절대 지어내지 말고 "제공된 문서에서 정보를 찾을 수 없습니다."라고 답변하세요.
    답변은 구체적인 조치 내용(시정명령, 과징금 등)을 포함하여 논리적이고 간결하게 작성하세요.
    """

    # 3. 실제 유저의 질문과 검색된 문서 내용을 합치기
    user_prompt = f"""
    [문서 내용]
    {context}
    
    [질문]
    {query}
    
    최종 답변:
    """

    # 4. OpenAI API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,  # 답변이 매번 바뀌지 않고 사실에 기반하도록 창의성 억제(0)
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"🚨 답변 생성 중 오류 발생: {e}"
