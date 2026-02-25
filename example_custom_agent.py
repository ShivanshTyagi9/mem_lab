"""
example_custom_agent.py — How to attach AgentMem to YOUR own agent.

This shows the integration pattern if you're not using the built-in MemoryAgent.
Copy this pattern and adapt it to your existing agent code.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from memory_system import MemorySystem
from dotenv import load_dotenv
load_dotenv()


# ── Step 1: Define your LLM function ─────────────────────────────────────────
# This can be ANY LLM — Claude, GPT, local model, etc.
# It just needs to take a string prompt and return a string response.

def my_llm(prompt: str) -> str:
    """Example using OpenAI. Replace model if needed."""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can change to gpt-4o or other available models
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7,
    )

    return response.choices[0].message.content


# ── Step 2: Initialize MemorySystem ───────────────────────────────────────────

mem = MemorySystem(llm_fn=my_llm)


# ── Step 3: PLUG at session start ─────────────────────────────────────────────

print("=== PLUG ===")
summary = mem.plug("examples/demo_user.agentmem")
# If file doesn't exist, starts with empty memory — that's fine.


# ── Step 4: Add some initial memories manually (optional) ─────────────────────
# This simulates a user who already has a history

if summary['semantic'] == 0:
    print("\nAdding initial memories for demo...")
    mem.add_memory_directly("The user's name is Aryan and they are building an AI agent framework", "semantic")
    mem.add_memory_directly("Aryan is based in India and prefers concise technical answers", "semantic")
    mem.add_memory_directly("Aryan prefers Python for all backend work", "procedural", "procedural/code_requests")
    mem.add_memory_directly("First session: Aryan discussed plug-and-play memory architecture for agents", "episodic")
    print(f"Memory stats: {mem.stats()}")


# ── Step 5: PLAY — use memory in your agent loop ──────────────────────────────

print("\n=== PLAY ===")
print("Demonstrating retrieval...\n")

queries = [
    "What do you know about the user?",
    "What kind of code should I write for them?",
    "What have we worked on before?",
]

for query in queries:
    print(f"Query: {query}")
    context = mem.retrieve(query)
    print(f"Retrieved context:\n{context}\n")
    print("-" * 40)


# ── Step 6: Simulate a conversation that generates new memories ───────────────

print("\n=== SIMULATING NEW SESSION ===")
fake_conversation = [
    {"role": "user", "content": "I just switched from building the memory system to working on the CLI interface"},
    {"role": "assistant", "content": "That's great progress! The CLI will make the system much more accessible."},
    {"role": "user", "content": "Yeah, I want it to support plug, inspect, and eject commands"},
    {"role": "assistant", "content": "Good structure. Three clean commands that map directly to the memory lifecycle."},
    {"role": "user", "content": "Also I realized I want to use networkx for the graph layer"},
    {"role": "assistant", "content": "Good choice. NetworkX is lightweight and perfect for this scale."},
]

new_ids = mem.update(fake_conversation)
print(f"New memory nodes created: {new_ids}")


# ── Step 7: EJECT at session end ──────────────────────────────────────────────

print("\n=== EJECT ===")
saved_path = mem.eject("examples/demo_user.agentmem")
print(f"Memory ejected to: {saved_path}")
print("The .agentmem file can now be stored, shared, or re-plugged into any agent.")
