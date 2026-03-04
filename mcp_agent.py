import uuid
import time
from typing import List

from client import connect_to_mcp
from llm import chat_completion


class MCPAgent:
    """
    Agent that uses a remote Memory Capsule through MCP.
    """

    def __init__(
        self,
        name: str,
        mcp_url: str = "http://localhost:8000",
        mode: str = "rw",
        top_k: int = 6,
    ):

        self.name = name
        self.session_id = str(uuid.uuid4())
        self.top_k = top_k

        print(f"Connecting to MCP memory server as {name}...")

        self.memory = connect_to_mcp(
            base_url=mcp_url,
            agent_id=name,
            mode=mode,
        )

        print("Memory connected")

        self.conversation_history: List[dict] = []

    # -------------------------
    # MEMORY RETRIEVAL
    # -------------------------

    def retrieve_context(self, query: str):

        try:
            res = self.memory.query_with_context(query, top_k=self.top_k)
            memories = res["results"]
        except Exception:
            return []

        context = []

        for m in memories:
            text = m.get("content") or m.get("text")
            if text:
                context.append(text)

        return context

    # -------------------------
    # BUILD PROMPT
    # -------------------------

    def build_prompt(self, user_message: str):

        retrieved = self.retrieve_context(user_message)

        context_block = "\n".join(
            f"- {m}" for m in retrieved
        )

        system_prompt = f"""
You are {self.name}, an AI agent with evolving memory.

Relevant memories:
{context_block}

Use them if helpful.
"""

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        messages += self.conversation_history
        messages.append({"role": "user", "content": user_message})

        return messages

    # -------------------------
    # GENERATE RESPONSE
    # -------------------------

    def think(self, user_message: str):

        messages = self.build_prompt(user_message)

        response = chat_completion(messages)

        return response

    # -------------------------
    # SAVE MEMORY
    # -------------------------

    def remember(self, user_message: str, response: str):

        memory_text = f"""
User: {user_message}
Agent: {response}
"""

        try:
            self.memory.ingest(memory_text)
        except Exception as e:
            print("Memory ingest failed:", e)

    # -------------------------
    # MAIN STEP
    # -------------------------

    def step(self, user_message: str):

        response = self.think(user_message)

        self.conversation_history.append(
            {"role": "user", "content": user_message}
        )

        self.conversation_history.append(
            {"role": "assistant", "content": response}
        )

        self.remember(user_message, response)

        return response

    # -------------------------
    # EXPORT MEMORY
    # -------------------------

    def export_memory(self):

        try:
            res = self.memory.export_snapshot()
            print("Memory exported:", res)
        except Exception as e:
            print("Export failed:", e)

    # -------------------------
    # CLOSE SESSION
    # -------------------------

    def shutdown(self):

        try:
            self.memory.disconnect()
        except Exception:
            pass

        print("Agent disconnected from memory")