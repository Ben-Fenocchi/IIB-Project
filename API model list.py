from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

models = client.models.list()

print("Available models:\n")
for m in models.data:
    print(m.id)


def try_model(model_name: str) -> None:
    print(f"\n--- Testing model: {model_name} ---")
    try:
        r = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with JSON: {\"ok\": true}"}],
            response_format={"type": "json_object"},
        )
        print("SUCCESS")
        print("Response:", r.choices[0].message.content)
    except Exception as e:
        print("FAILED")
        print("Error:", repr(e))

if __name__ == "__main__":
    try_model("gpt-5-mini")
    #try_model("gpt-5-nano")
    #try_model("gpt-5-nano-2025-08-07")
