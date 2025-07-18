# https://ai.google.dev/gemini-api/docs/quickstart
from google import genai


class LLM:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def generate_response(self, prompt):
        response = self.client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return response.text
