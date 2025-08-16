"""Conversational agent with memory."""
from typing import List, Dict

class ConversationMemory:
    def __init__(self, max_turns=5):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def clear(self):
        self.history = []

class RAGAgent:
    def __init__(self, retriever, system_prompt="You are a helpful assistant.", memory_k=5):
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.memory = ConversationMemory(max_turns=memory_k)

    def build_prompt(self, query: str) -> str:
        context = self.retriever.build_context(query)
        history_str = ""
        for msg in self.memory.get_history():
            history_str += f"{msg['role'].upper()}: {msg['content']}\n"
        prompt = f"""{self.system_prompt}

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {query}

Answer based on the context provided. If the context doesn't contain relevant information, say so."""
        return prompt

    def chat(self, query: str) -> str:
        self.memory.add("user", query)
        prompt = self.build_prompt(query)
        response = f"Based on the retrieved context, here is my answer to: {query}"
        self.memory.add("assistant", response)
        return response

    def reset(self):
        self.memory.clear()
