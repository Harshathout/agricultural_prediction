from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
print("KEY:", os.getenv("GEMINI_API_KEY")[:10])

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

print(model.generate_content("Hello").text)
