from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # Load your .env variables

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
)

print(response.choices[0].message.content)
