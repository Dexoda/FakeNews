from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

PROMPT = """
Проверь следующий текст на признаки фейк-ньюс.
Никогда не выдавай предположения или догадки как факты.
Если не можешь проверить информацию напрямую, напиши: "Не могу проверить."
Не интерпретируй и не перефразируй текст без запроса.
Любую непроверенную часть помечай: [Непроверено], [Домысел], [Спекуляция].
Если есть непроверенные части, пометь весь ответ.
Ответь одним из вариантов: "Фейк", "Реально", "Требует проверки".
Дай краткое пояснение.
Текст:
"""

def ai_fact_check(text):
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://github.com/Dexoda/FakeNews",
            "X-Title": "FakeNewsDetector",
        },
        model="qwen/qwen3-32b:free",
        messages=[
            {"role": "user", "content": PROMPT + text}
        ]
    )
    return completion.choices[0].message.content
