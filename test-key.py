from openai import OpenAI
import os

key = os.getenv("OPENAI_API_KEY")
if not key:
    raise SystemExit("OPENAI_API_KEY not set in environment.")

client = OpenAI(api_key=key)

try:
    models = client.models.list()
    print("API key is VALID ✅. Found models:", len(models.data))
except Exception as e:
    print("API key test FAILED ❌")
    print(e)
