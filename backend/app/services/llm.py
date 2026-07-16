from groq import Groq
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        settings.validate_keys()
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    def generate_answer(self, messages: list[dict]) -> str:
        """
        Calls Groq with the constructed context, history, and message structure.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.0,  # Zero temperature for deterministic RAG grounding
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate completion from Groq: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")
