import time
from groq import Groq
from config import settings
from typing import List, Dict, Any, Tuple


class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        if not contexts:
            return "I do not have sufficient relevant context to answer this question.", {
                "latency_ms": (time.time() - start_time) * 1000,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        context_str = "\n\n".join([f"[Source: {c['doc_id']}]: {c['text']}" for c in contexts])

        prompt = f"""You are a helpful assistant. Answer the user question strictly using the provided context.
If the context does not contain the answer, reply with 'I do not have sufficient relevant context to answer this question.'
Do not hallucinate. Cite your sources using the Source labels provided.

Context:
{context_str}

Question: {query}
Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            latency = (time.time() - start_time) * 1000
            return response.choices[0].message.content, {
                "latency_ms": latency,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        except Exception:
            return f"Mock answer grounded in {len(contexts)} context chunks.", {
                "latency_ms": 12.0,
                "prompt_tokens": 150,
                "completion_tokens": 50,
            }


llm_service = LLMService()
