"""Tests for conversational agent."""
import unittest
from src.agent import ConversationMemory, RAGAgent

class TestMemory(unittest.TestCase):
    def test_add_and_retrieve(self):
        mem = ConversationMemory(max_turns=3)
        mem.add("user", "hello")
        mem.add("assistant", "hi")
        self.assertEqual(len(mem.get_history()), 2)

    def test_max_turns(self):
        mem = ConversationMemory(max_turns=2)
        for i in range(10):
            mem.add("user", f"msg {i}")
        self.assertLessEqual(len(mem.get_history()), 4)

if __name__ == "__main__":
    unittest.main()
