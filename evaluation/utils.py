import os
import requests
import time
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval_utils")

# Load environment variables
# Check if there is an env file in backend and load it
backend_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))
if os.path.exists(backend_env_path):
    load_dotenv(backend_env_path)
    logger.info(f"Loaded environment variables from backend env: {backend_env_path}")
else:
    load_dotenv()

class ChatbotClient:
    """
    Client wrapper for interacting with the chatbot FastAPI endpoint.
    Configurable to support various input/output schemas.
    """
    def __init__(
        self, 
        endpoint_url="http://localhost:8000/chat", 
        request_field="message",  # field name for sending user queries
        response_field="answer",  # field name for reading chatbot answer
        sources_field="sources"   # field name for reading retrieved source chunks
    ):
        self.endpoint_url = endpoint_url
        self.request_field = request_field
        self.response_field = response_field
        self.sources_field = sources_field
        logger.info(
            f"Initialized ChatbotClient pointing to {endpoint_url} "
            f"(ReqField='{request_field}', RespField='{response_field}', SourcesField='{sources_field}')"
        )
        
    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Sends a POST request to the configured chatbot endpoint.
        Returns:
            dict with:
                - answer (str): chatbot response
                - sources (list): retrieved source dictionaries
                - latency (float): request duration in seconds
                - success (bool): request succeeded or not
                - error (str): error details if any
        """
        # Build payload dynamically based on configured input field
        payload = {
            self.request_field: question
        }
        
        # If targeting our specific backend, support custom parameters
        if self.request_field == "message":
            payload["top_k"] = top_k
            # Unique session to isolate evaluation turns from one another
            payload["session_id"] = f"eval-{int(time.time() * 1000)}"
            
        start_time = time.time()
        try:
            logger.debug(f"Sending POST request to {self.endpoint_url} with payload {payload}")
            response = requests.post(self.endpoint_url, json=payload, timeout=20)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                resp_data = response.json()
                answer = resp_data.get(self.response_field, "")
                sources = resp_data.get(self.sources_field, [])
                return {
                    "answer": answer,
                    "sources": sources,
                    "latency": latency,
                    "success": True,
                    "error": None
                }
            else:
                return {
                    "answer": "",
                    "sources": [],
                    "latency": latency,
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Failed to query chatbot endpoint: {e}")
            return {
                "answer": "",
                "sources": [],
                "latency": latency,
                "success": False,
                "error": str(e)
            }
