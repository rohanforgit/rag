import time
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, max_sessions: int = 500, ttl_seconds: int = 3600):
        self.sessions = OrderedDict()
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds

    def _prune(self):
        """Prunes expired sessions or old ones if capacity is exceeded."""
        now = time.time()
        # Remove expired sessions
        expired_keys = [k for k, v in self.sessions.items() if now - v['last_accessed'] > self.ttl_seconds]
        for k in expired_keys:
            self.sessions.pop(k, None)
            
        # Limit total size
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)

    def get_history(self, session_id: str) -> list[dict]:
        """Retrieves history for session_id. Updates access time."""
        self._prune()
        if session_id in self.sessions:
            self.sessions[session_id]['last_accessed'] = time.time()
            return self.sessions[session_id]['messages']
        return []

    def add_message(self, session_id: str, role: str, content: str):
        """Appends a message to the history of session_id."""
        self._prune()
        now = time.time()
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'messages': [],
                'created_at': now
            }
            
        self.sessions[session_id]['last_accessed'] = now
        self.sessions[session_id]['messages'].append({"role": role, "content": content})
        
        # Limit history to 16 messages (8 turns) to avoid prompt bloat
        if len(self.sessions[session_id]['messages']) > 16:
            self.sessions[session_id]['messages'] = self.sessions[session_id]['messages'][-16:]
            
    def clear_session(self, session_id: str):
        """Clears a session manually."""
        self.sessions.pop(session_id, None)
