import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env at project root
load_dotenv()

# Initialize OpenAI client (expects OPENAI_API_KEY in env)
client = OpenAI()

# Load CV text
with open("data/cv_1.txt", "r", encoding="utf-8") as f:
    cv_text = f.read()

# Load schema (for the prompt only, not validation yet)
with open("schemas/cv_schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

# Prompt the LLM
prompt = f"""
You are an information extraction system.

Extract the following fields from the CV text below.
Return ONLY valid JSON.
Do not add explanations.

Schema:
{json.dumps(schema, indent=2)}

CV:
{cv_text}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You extract structured data."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

raw_output = response.choices[0].message.content

print("\nRAW MODEL OUTPUT:\n")
print(raw_output)

# Try to parse JSON (this may fail — that's expected today)
try:
    data = json.loads(raw_output)
    print("\nPARSED JSON:\n")
    print(json.dumps(data, indent=2))
except json.JSONDecodeError as e:
    print("\nJSON PARSE ERROR:")
    print(e)
