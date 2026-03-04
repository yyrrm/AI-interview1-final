from dotenv import load_dotenv
import os
from openai import OpenAI

# .env 파일 로드
load_dotenv()

# OpenAI 클라이언트 생성
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def make_question(answer_text, topic="면접"):
    prompt = f"""
너는 실제 기업 면접에서 사용되는 질문을 생성하는 전문 면접관 AI이다. 목표는 지원자의 역량을 폭넓게 확인하는 것이다.

[면접 목적]
지원자의 역량, 사고력, 기술, 의사소통 능력을 자연스럽게 평가한다.

[이전 지원자 답변]
{answer_text}

[질문 생성 규칙]
1. 이미 했던 질문과 의미가 겹치는 질문은 금지
2. 지나치게 공격적이거나 부정적인 질문은 피할 것
3. 실제 면접에서 사용 가능한 현실적인 질문일 것
4. 한 문장에서 두 문장으로 출력할 것
5. 질문만 출력할 것 (부가 설명 금지).
6. 아래 카테고리 중 하나를 선택해서 질문하되, 가능한 다양하게:
   - 프로젝트/경험(구체 사례)
   - 문제 해결/트러블슈팅
   - 팀 협업/갈등 해결
   - 기술 선택 이유/설계 판단
   - 강점/약점과 개선 노력
   - 압박 질문(근거 요구)

이전 답변을 바탕으로 다음 면접 질문을 생성하라.


[면접 주제]
{topic}

[지원자 답변]
{answer_text}

조건:
- 질문만 출력
- 한 문장에서 두 문장
- 너무 길지 않게
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()
