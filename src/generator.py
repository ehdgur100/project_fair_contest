# src/generator.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# 보안을 위해 숨겨둔 .env 파일에서 OpenAI API 키를 가져와 연결합니다.
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_answer(query: str, retrieved_chunks: list) -> str:
    """검색된 5개의 청크를 바탕으로 GPT-4o-mini가 실제 답변을 생성합니다."""

    if not retrieved_chunks:
        return "검색된 관련 문서가 없어 답변할 수 없습니다."

    # 1. 검색 엔진이 찾아온 5개의 텍스트를 하나의 아주 긴 글로 합칩니다.
    context = "\n\n---\n\n".join(retrieved_chunks)

    # 2. [가장 중요한 프롬프트] AI를 가스라이팅(?)하는 지시사항입니다.
    # 💡 지금은 서술형으로 되어있습니다. 이 부분을 "원문에서 딱 한 줄만 발췌해라!"로 고치면 점수가 오릅니다.
    system_prompt = """
    당신은 대한민국 공정거래위원회 의결서를 분석하는 '법률 AI 어시스턴트'입니다.
    반드시 아래 제공된 [문서 내용]만을 근거로 하여 질문에 답변하세요.
    문서 내용에 없는 정보는 절대 지어내지 말고 "제공된 문서에서 정보를 찾을 수 없습니다."라고 답변하세요.
    답변은 구체적인 조치 내용(시정명령, 과징금 등)을 포함하여 논리적이고 간결하게 작성하세요.
    """

    user_prompt = f"""
    [문서 내용]
    {context}
    
    [질문]
    {query}
    
    최종 답변:
    """

    # 3. GPT 모델을 호출하여 대답을 듣습니다.
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,  # 0으로 설정하면 창의성을 버리고 기계처럼 딱딱하고 일관되게 대답합니다.
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"🚨 답변 생성 중 오류 발생: {e}"
