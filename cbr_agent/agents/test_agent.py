from google import genai

client = genai.Client(api_key="")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Calculate the sum of exponentials of first 3 Fibonacci numbers",
)

print(response.text)