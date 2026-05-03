import os
import requests
import json
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()

# DeepSeek's key and URL
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

class RateLimitError(Exception):
    pass

class OpenRouterHarness: 
    # Using the deepseek-v4-flash model
    def __init__(self, default_model="deepseek-v4-flash"):
        self.default_model = default_model
        self.headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }

    # Keep a gentle backoff just in case of server load
    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(RateLimitError)
    )
    def generate_response_with_tokens(self, system_prompt, user_prompt, model=None, temperature=0.7):
        """Returns a tuple: (content_string, total_tokens_used)"""
        payload = {
            "model": model or self.default_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        response = requests.post(DEEPSEEK_URL, headers=self.headers, data=json.dumps(payload))
        
        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded (429).")
        response.raise_for_status() 
        
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        tokens = response_json.get('usage', {}).get('total_tokens', 0)
        
        return content, tokens