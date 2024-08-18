import os
import json
from collections import deque

class CacheManager:
    def __init__(self, repo_name, max_history=100):  # Increased max_history for global chat
        self.repo_name = repo_name
        self.max_history = max_history
        self.base_cache_dir = os.path.join('cache', self.repo_name)
        self.conversation_dir = os.path.join(self.base_cache_dir, 'conversations')
        self.ai_chat_file = os.path.join(self.base_cache_dir, 'ai_chat.jsonl')
        os.makedirs(self.conversation_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.ai_chat_file), exist_ok=True)

    def get_conversation_history(self, user_id=None, is_ai_chat=False):
        if is_ai_chat:
            return self._get_ai_chat_history()
        else:
            return self._get_user_conversation_history(user_id)

    def _get_ai_chat_history(self):
        history = deque(maxlen=self.max_history)
        if os.path.exists(self.ai_chat_file):
            with open(self.ai_chat_file, 'r') as f:
                for line in f:
                    history.append(json.loads(line.strip()))
        return list(history)

    def _get_user_conversation_history(self, user_id):
        file_path = os.path.join(self.conversation_dir, f"{user_id}.jsonl")
        history = deque(maxlen=self.max_history)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    history.append(json.loads(line.strip()))
        return list(history)

    def append_to_conversation(self, user_id, message, is_ai_chat=False):
        if is_ai_chat:
            self._append_to_ai_chat(message)
        else:
            self._append_to_user_conversation(user_id, message)

    def _append_to_ai_chat(self, message):
        history = self._get_ai_chat_history()
        history.append(message)
        if len(history) > self.max_history:
            history = history[-self.max_history:]
        with open(self.ai_chat_file, 'w') as f:
            for item in history:
                f.write(json.dumps(item) + '\n')

    def _append_to_user_conversation(self, user_id, message):
        history = self._get_user_conversation_history(user_id)
        history.append(message)
        if len(history) > self.max_history:
            history = history[-self.max_history:]
        file_path = os.path.join(self.conversation_dir, f"{user_id}.jsonl")
        with open(file_path, 'w') as f:
            for item in history:
                f.write(json.dumps(item) + '\n')

    def clear_conversation(self, user_id=None, is_ai_chat=False):
        if is_ai_chat:
            if os.path.exists(self.ai_chat_file):
                os.remove(self.ai_chat_file)
        else:
            file_path = os.path.join(self.conversation_dir, f"{user_id}.jsonl")
            if os.path.exists(file_path):
                os.remove(file_path)

    def get_cache_dir(self, cache_type):
        cache_dir = os.path.join(self.base_cache_dir, cache_type)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir