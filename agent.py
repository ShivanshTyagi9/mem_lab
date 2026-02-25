"""
agent.py — Ready-to-use Claude agent with AgentMem attached.

This is the simplest way to use the system:
    agent = MemoryAgent(api_key="...", memory_path="aryan.agentmem")
    agent.plug()
    response = agent.chat("Hello, what do you remember about me?")
    agent.eject()

The agent handles:
  - Memory retrieval on every turn
  - Memory extraction at eject time
  - Full conversation history management
"""

import os
from typing import Optional
from memory_system import MemorySystem

class MemoryAgent:
    """
    Claude agent with 3-layer plug-and-play memory.
    
    Can be used standalone or as a reference for integrating
    MemorySystem into your own agent.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        memory_path: str = "memory.agentmem",
        model: str = "gpt-4o-mini",
        system_persona: str = "You are a helpful, personalized AI assistant.",
    ):
        self.memory_path = memory_path
        self.model = model
        self.system_persona = system_persona
        self._conversation: list[dict] = []
        self._memory: Optional[MemorySystem] = None

        # setup Anthropic client
        try:
            from openai import OpenAI
            import os

            self._client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY")
            )

        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        # create memory system with LLM function
        self._memory = MemorySystem(llm_fn=self._llm_call)

    # ── Public Interface ──────────────────────────────────────────────────────

    def plug(self) -> dict:
        """Load memory from file. Call this before starting a session."""
        return self._memory.plug(self.memory_path)

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        Memory is automatically retrieved and injected each turn.
        """
        # retrieve relevant memory for this query
        memory_context = self._memory.retrieve(user_message)

        # build system prompt with memory injected
        system = self._build_system_prompt(memory_context)

        # add to conversation
        self._conversation.append({"role": "user", "content": user_message})

        # call Claude
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                *self._conversation
            ],
            max_tokens=1000,
        )

        assistant_message = response.choices[0].message.content

        self._conversation.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def eject(self) -> str:
        """
        Extract memories from this session, save, and return updated memory file.
        Call this at end of session.
        """
        if self._conversation:
            self._memory.update(self._conversation)
        return self._memory.eject(self.memory_path)

    def stats(self) -> dict:
        """Return current memory statistics."""
        return self._memory.stats()

    def add_memory(self, content: str, memory_type: str = "semantic") -> str:
        """Manually add a memory without LLM extraction."""
        return self._memory.add_memory_directly(content, memory_type)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _llm_call(self, prompt):
    # Auto-wrap string prompts into message format
        if isinstance(prompt, str):
            messages = [
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt  # already structured

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )

        return response.choices[0].message.content

    def _build_system_prompt(self, memory_context: str) -> str:
        parts = [self.system_persona]
        if memory_context:
            parts.append("\n" + memory_context)
        parts.append(
            "\nUse your memory naturally — don't announce that you're reading from memory. "
            "Just behave as if you genuinely know the user."
        )
        return "\n".join(parts)
