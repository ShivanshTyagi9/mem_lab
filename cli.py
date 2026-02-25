#!/usr/bin/env python3
"""
cli.py — Command line interface for AgentMem.

Commands:
  python cli.py chat --memory aryan.agentmem       # start a chat session
  python cli.py inspect --memory aryan.agentmem    # inspect memory contents
  python cli.py add --memory aryan.agentmem        # manually add a memory
"""

import sys
import os
import argparse
import json
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_system import  MemorySystem
from agent import MemoryAgent


def cmd_chat(args):
    """Start an interactive chat session with memory."""
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Provide --api-key or set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  AgentMem Chat Session")
    print(f"  Memory: {args.memory}")
    print(f"{'='*50}")
    print("  Type 'exit' to end session and save memory")
    print("  Type 'stats' to see memory statistics")
    print(f"{'='*50}\n")

    agent = MemoryAgent(
        api_key=api_key,
        memory_path=args.memory,
        system_persona=args.persona or "You are a helpful, personalized AI assistant.",
    )

    # PLUG
    agent.plug()

    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                break

            if user_input.lower() == "stats":
                stats = agent.stats()
                print(f"\n[Memory Stats]")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
                print()
                continue

            response = agent.chat(user_input)
            print(f"\nAgent: {response}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving memory...")

    # EJECT
    print("\nExtracting and saving memories...")
    saved_path = agent.eject()
    print(f"Memory saved to: {saved_path}")
    stats = agent.stats()
    print(f"Session added {stats['session_nodes_added']} new memory nodes")


def cmd_inspect(args):
    """Inspect the contents of a .agentmem file without running a session."""

    # create a memory system with no LLM (read-only)
    mem = MemorySystem(llm_fn=None)
    summary = mem.plug(args.memory)

    print(f"\n{'='*50}")
    print(f"  Memory Inspection: {args.memory}")
    print(f"{'='*50}")
    print(f"  Semantic nodes:   {summary['semantic']}")
    print(f"  Episodic nodes:   {summary['episodic']}")
    print(f"  Procedural nodes: {summary['procedural']}")
    print(f"  Total nodes:      {mem.stats()['total_nodes']}")
    print(f"{'='*50}\n")

    # show all nodes
    all_ids = mem._graph.all_node_ids()
    by_type = {"semantic": [], "episodic": [], "procedural": []}

    for nid in all_ids:
        node = mem._graph.get_node(nid)
        if node:
            by_type.get(node.memory_type, []).append(node)

    for mtype, nodes in by_type.items():
        if nodes:
            print(f"[{mtype.upper()}]")
            for node in sorted(nodes, key=lambda n: n.timestamp, reverse=True):
                conf = f"({node.confidence:.1f})" if node.confidence < 0.8 else ""
                print(f"  • {node.content} {conf}")
                for edge in node.edges:
                    print(f"    └─ {edge.type} → {edge.to}")
            print()

    # cleanup without saving
    import shutil
    if mem._work_dir:
        shutil.rmtree(mem._work_dir, ignore_errors=True)


def cmd_add(args):
    """Manually add a memory to a .agentmem file."""
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    mem = MemorySystem(llm_fn=None)  # no LLM needed for manual add
    mem.plug(args.memory)

    content = args.content or input("Memory content: ").strip()
    memory_type = args.type or "semantic"

    node_id = mem.add_memory_directly(content, memory_type)
    print(f"Added memory: {node_id}")
    print(f"Content: {content}")

    mem.eject(args.memory)
    print(f"Saved to {args.memory}")


def main():
    parser = argparse.ArgumentParser(
        description="AgentMem — plug-and-play memory for AI agents"
    )
    subparsers = parser.add_subparsers(dest="command")

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Start a chat session with memory")
    chat_parser.add_argument("--memory", default="memory.agentmem", help="Path to .agentmem file")
    chat_parser.add_argument("--api-key", help="Anthropic API key")
    chat_parser.add_argument("--persona", help="Agent system persona")

    # inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect memory file contents")
    inspect_parser.add_argument("--memory", default="memory.agentmem", help="Path to .agentmem file")

    # add command
    add_parser = subparsers.add_parser("add", help="Manually add a memory")
    add_parser.add_argument("--memory", default="memory.agentmem", help="Path to .agentmem file")
    add_parser.add_argument("--content", help="Memory content")
    add_parser.add_argument("--type", choices=["semantic", "episodic", "procedural"], default="semantic")
    add_parser.add_argument("--api-key", help="Anthropic API key")

    args = parser.parse_args()

    if args.command == "chat":
        cmd_chat(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "add":
        cmd_add(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
