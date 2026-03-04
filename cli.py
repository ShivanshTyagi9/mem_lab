#!/usr/bin/env python3
"""
cli.py — uMemory Chat CLI

Usage:
  python cli.py                          # start fresh session
  python cli.py --capsule alex.umc       # resume from capsule
  python cli.py --capsule alex.umc --name alex  # resume with custom name
  python cli.py --name myagent           # fresh session with name

Commands (type during chat):
  /help          show all commands
  /memory        show what the agent currently remembers (relevant to last message)
  /stats         show memory graph statistics
  /branch BRANCH dump all nodes in a branch (People, Tasks, Technical, etc.)
  /history TOPIC show current + superseded memories for a topic
  /consolidate   merge episodic memories into stable facts
  /decay         prune weak memories
  /save [path]   save capsule to file
  /load path     load a different capsule mid-session
  /clear         clear conversation history (keep memory)
  /verbose       toggle verbose ingestion output
  /quit          save and exit
"""

import os
import sys
import argparse
import textwrap
from typing import Optional

# ── terminal colours ──────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
CYAN   = "\033[36m"
WHITE  = "\033[37m"
GREY   = "\033[90m"

def c(text, *codes):
    return "".join(codes) + str(text) + RESET

def hr(char="─", width=60, colour=GREY):
    return c(char * width, colour)


# ── banner ────────────────────────────────────────────────────────────────────

BANNER = f"""
{c('┌─────────────────────────────────────────┐', CYAN)}
{c('│', CYAN)}  {c('u', BOLD + CYAN)}{c('Memory', CYAN)}  {c('plug-and-play agent memory', DIM)}    {c('│', CYAN)}
{c('└─────────────────────────────────────────┘', CYAN)}
{c('Type /help for commands', GREY)}
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def wrap(text: str, width: int = 72, indent: str = "") -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if line.strip() == "":
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(line, width, subsequent_indent=indent))
    return "\n".join(wrapped)


def print_memory_block(memories, label="Relevant memories"):
    if not memories:
        print(c(f"  (no {label.lower()} found)", GREY))
        return
    print(c(f"  {label}:", DIM))
    for m in memories:
        print(c("  · ", GREY) + c(m, WHITE))


def print_stats(stats: dict):
    print(c("  Memory stats:", DIM))
    print(f"  {c('total nodes', GREY)}   {c(stats['total_nodes'], CYAN)}")
    print(f"  {c('latest', GREY)}        {c(stats['latest_nodes'], GREEN)}")
    print(f"  {c('superseded', GREY)}    {c(stats['superseded'], YELLOW)}")
    print(f"  {c('total edges', GREY)}   {c(stats['total_edges'], CYAN)}")
    print(f"  {c('ingestions', GREY)}    {c(stats.get('ingest_count', '?'), CYAN)}")
    print()

    by_type = stats.get("by_type", {})
    if by_type:
        print(c("  by type:", DIM))
        for t, n in sorted(by_type.items()):
            bar = "█" * min(n, 20)
            print(f"    {c(t.ljust(12), GREY)} {c(bar, CYAN)} {c(n, WHITE)}")
    print()

    branches = stats.get("branches", {})
    if branches:
        print(c("  by branch:", DIM))
        for branch, count in sorted(branches.items(), key=lambda x: -x[1]):
            print(f"    {c(branch.ljust(14), GREY)} {c(count, WHITE)}")


def print_help():
    commands = [
        ("/help",              "Show this help"),
        ("/memory [topic]",    "Show memories relevant to topic (or last message)"),
        ("/stats",             "Memory graph statistics"),
        ("/branch BRANCH",     "Dump all nodes in a branch"),
        ("/history TOPIC",     "Current + superseded memories for a topic"),
        ("/timeline TOPIC",    "Full version history with timestamps for a topic"),
        ("/tasks",             "Show all pending tasks and reminders"),
        ("/done ID",           "Mark a task as done by its short ID"),
        ("/export-graph",      "Export graph JSON for visualization"),
        ("/consolidate",       "Merge episodes into stable derived facts"),
        ("/decay [threshold]", "Prune weak memories (default threshold: 0.05)"),
        ("/save [path]",       "Save capsule (default: <name>.umc)"),
        ("/load path",         "Load a capsule file"),
        ("/clear",             "Clear conversation history (keep memory)"),
        ("/verbose",           "Toggle verbose ingestion logging"),
        ("/quit",              "Save and exit"),
    ]
    print()
    print(c("  Commands:", BOLD))
    for cmd, desc in commands:
        print(f"  {c(cmd.ljust(24), CYAN)} {c(desc, GREY)}")
    print()
    branches = ["People", "Tasks", "Technical", "Preferences", "Events", "General"]
    print(c("  Branches:", BOLD))
    print(f"  {c(', '.join(branches), GREY)}")
    print()


# ── main CLI class ────────────────────────────────────────────────────────────

class MemoryCLI:

    def __init__(self, capsule_path: Optional[str] = None, name: str = "agent", debug: bool = False):
        from agent import AgentSession
        from memory_socket import MemorySocket, AccessMode

        print(BANNER)

        if capsule_path and os.path.exists(capsule_path):
            print(c(f"  Loading capsule: {capsule_path}", GREY))
            self.agent = AgentSession.resume(capsule_path)
            print(c(f"  ✓ {len(self.agent.memory.graph.nodes)} memories restored", GREEN))
        else:
            if capsule_path:
                print(c(f"  No capsule at '{capsule_path}' — starting fresh.", YELLOW))
            self.agent = AgentSession(name=name)
            print(c(f"  Fresh session: '{name}'", GREY))

        self.verbose      = debug  # --debug flag sets verbose from start
        self.last_message = ""
        self.capsule_path = capsule_path or f"{self.agent.name}.umc"

        print(hr())
        print()

    def run(self):
        while True:
            try:
                raw = input(c("you  › ", BOLD + BLUE)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self._cmd_quit()
                break

            if not raw:
                continue

            if raw.startswith("/"):
                self._handle_command(raw)
            else:
                self._handle_message(raw)

    # ── message handling ──────────────────────────────────────────────────

    def _handle_message(self, text: str):
        self.last_message = text
        print()

        # Show a brief "thinking" indicator
        print(c("  …", GREY), end="\r")

        try:
            reply = self.agent.chat(text, verbose=self.verbose)
        except Exception as e:
            print(c(f"  Error: {e}", RED))
            return

        # Clear the thinking indicator line
        print(" " * 40, end="\r")

        # Print reply
        print(c("agent", BOLD + GREEN) + c(" › ", GREY))
        print()
        for line in wrap(reply, width=70, indent="  ").split("\n"):
            print(f"  {line}")
        print()
        print(hr())
        print()

    # ── command dispatch ──────────────────────────────────────────────────

    def _handle_command(self, raw: str):
        parts = raw.split(None, 1)
        cmd   = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ""

        print()

        dispatch = {
            "/help":        lambda: print_help(),
            "/stats":       self._cmd_stats,
            "/consolidate": self._cmd_consolidate,
            "/verbose":     self._cmd_verbose,
            "/clear":       self._cmd_clear,
            "/quit":        self._cmd_quit,
            "/q":           self._cmd_quit,
        }

        if cmd in dispatch:
            dispatch[cmd]()
        elif cmd == "/memory":
            self._cmd_memory(arg or self.last_message)
        elif cmd == "/history":
            self._cmd_history(arg or self.last_message)
        elif cmd == "/timeline":
            self._cmd_timeline(arg or self.last_message)
        elif cmd == "/tasks":
            self._cmd_tasks()
        elif cmd == "/done":
            self._cmd_done(arg)
        elif cmd == "/export-graph":
            self._cmd_export_graph(arg)
        elif cmd == "/branch":
            self._cmd_branch(arg)
        elif cmd == "/save":
            self._cmd_save(arg)
        elif cmd == "/load":
            self._cmd_load(arg)
        elif cmd == "/decay":
            self._cmd_decay(arg)
        else:
            print(c(f"  Unknown command: {cmd}. Type /help.", YELLOW))

        print()

    def _cmd_memory(self, topic: str):
        if not topic:
            print(c("  Usage: /memory <topic>", YELLOW))
            return
        memories = self.agent.memory.query(topic, top_k=8)
        print(c(f"  Memories for: '{topic}'", DIM))
        print()
        print_memory_block(memories, label="Current memories")

    def _cmd_history(self, topic: str):
        if not topic:
            print(c("  Usage: /history <topic>", YELLOW))
            return
        result = self.agent.memory.query_with_history(topic, top_k=8)
        print(c(f"  Memory history for: '{topic}'", DIM))
        print()
        print(c("  Current (latest truth):", GREEN))
        for m in result["current"]:
            print(c("  ✓ ", GREEN) + c(m, WHITE))
        if not result["current"]:
            print(c("  (none)", GREY))
        print()
        print(c("  History (superseded):", YELLOW))
        for m in result["history"]:
            print(c("  · ", YELLOW) + c(m, GREY))
        if not result["history"]:
            print(c("  (none)", GREY))

    def _cmd_timeline(self, topic: str):
        if not topic:
            print(c("  Usage: /timeline <topic>", YELLOW))
            return

        print(c(f"  Fact timeline for: '{topic}'", DIM))
        print()

        timelines = self.agent.memory._socket._capsule.query_timeline(topic, top_k=3)

        if not timelines:
            print(c("  No matching facts found.", GREY))
            return

        for tl in timelines:
            versions = tl.get("versions", [])
            if not versions:
                continue

            v_count = len(versions)
            print(c(f"  ● {tl['current_fact']}", WHITE + BOLD))
            if v_count > 1:
                print(c(f"    {v_count} versions over time:", DIM))
            print()

            for v in versions:
                vnum     = v["version"]
                is_cur   = v["is_current"]
                from_dt  = v["believed_from"]
                until_dt = v["believed_until"] or "present"
                dur      = v["belief_duration"]
                reason   = v.get("update_reason", "")
                content  = v["content"]

                marker  = c(f"  v{vnum}", GREEN if is_cur else YELLOW)
                date_range = c(f"{from_dt} → {until_dt}", GREY)
                dur_str    = c(f"({dur})", DIM)
                print(f"  {marker}  {date_range}  {dur_str}")
                print(f"      {c(content, WHITE if is_cur else GREY)}")
                if reason:
                    print(f"      {c('reason: ' + reason, DIM)}")
                print()

            print(hr())
            print()

    def _cmd_tasks(self):
        capsule = self.agent.memory._socket._capsule
        tasks   = capsule.get_pending_tasks()
        print(c("  Pending tasks:", DIM))
        print()
        if not tasks:
            print(c("  (no pending tasks)", GREY))
            return
        for task in tasks:
            short_id = task.id[:6]
            title    = task.task_title or task.content[:60]
            due      = c(f"  due {task.due_date}", YELLOW) if task.due_date else ""
            print(
                f"  {c('[' + short_id + ']', CYAN)}  "
                f"{c(title, WHITE)}{due}"
            )
            print(f"       {c(task.content, GREY)}")
            print()

    def _cmd_done(self, arg: str):
        if not arg:
            print(c("  Usage: /done <short-id>  (first 6 chars of task ID)", YELLOW))
            return
        capsule = self.agent.memory._socket._capsule
        # Match by prefix
        matched = [
            nid for nid in capsule.graph.nodes
            if nid.startswith(arg) and
            capsule.graph.nodes[nid].mem_type == "task"
        ]
        if not matched:
            print(c(f"  No task found with ID starting '{arg}'.", RED))
            return
        if len(matched) > 1:
            print(c(f"  Ambiguous ID — {len(matched)} tasks match. Use more characters.", YELLOW))
            return
        node = capsule.graph.nodes[matched[0]]
        capsule.mark_task_done(matched[0])
        print(c(f"  ✓ Done: {node.task_title or node.content[:60]}", GREEN))

    def _cmd_export_graph(self, arg: str):
        capsule = self.agent.memory._socket._capsule
        path    = arg.strip() or None
        try:
            out = capsule.export_graph_json(path)
            print(c(f"  ✓ Graph exported → {out}", GREEN))
            print(c(f"  Open memory_viz.html and load this file to visualize.", GREY))
        except Exception as e:
            print(c(f"  Export failed: {e}", RED))

    def _cmd_branch(self, branch: str):
        if not branch:
            print(c("  Usage: /branch <BranchName>", YELLOW))
            print(c("  Branches: People, Tasks, Technical, Preferences, Events, General", GREY))
            return

        # Case-insensitive match
        from models import TREE_BRANCHES
        matched = next((b for b in TREE_BRANCHES if b.lower() == branch.lower()), None)
        if not matched:
            print(c(f"  Unknown branch '{branch}'.", YELLOW))
            print(c(f"  Available: {', '.join(TREE_BRANCHES)}", GREY))
            return

        nodes = self.agent.memory.dump_branch(matched, include_superseded=True)
        if not nodes:
            print(c(f"  No memories in branch '{matched}'.", GREY))
            return

        print(c(f"  Branch: {matched}  ({len(nodes)} nodes)", DIM))
        print()
        for n in nodes:
            latest_marker = c("✓", GREEN) if n["is_latest"] else c("·", YELLOW)
            strength_bar  = "█" * int(n["strength"] * 10)
            print(
                f"  {latest_marker} "
                f"{c(n['type'].ljust(11), GREY)} "
                f"{c(strength_bar.ljust(10), CYAN + DIM)} "
                f"{c(n['content'][:70], WHITE)}"
            )

    def _cmd_stats(self):
        stats = self.agent.memory_stats()
        print_stats(stats)

    def _cmd_consolidate(self):
        print(c("  Consolidating episodic memories…", GREY))
        n = self.agent.memory.consolidate(verbose=True)
        if n == 0:
            print(c("  Nothing to consolidate (need ≥3 episodes in a branch).", GREY))
        else:
            print(c(f"  ✓ {n} group(s) consolidated.", GREEN))

    def _cmd_decay(self, arg: str):
        try:
            threshold = float(arg) if arg else 0.05
        except ValueError:
            print(c(f"  Invalid threshold: '{arg}'", YELLOW))
            return
        print(c(f"  Pruning memories below strength {threshold}…", GREY))
        pruned = self.agent.memory.decay(threshold=threshold, verbose=True)
        if pruned == 0:
            print(c("  Nothing pruned.", GREY))
        else:
            print(c(f"  ✓ {pruned} nodes removed.", GREEN))

    def _cmd_save(self, path: str):
        save_path = path or self.capsule_path
        try:
            actual = self.agent.save(save_path)
            self.capsule_path = actual
            size_kb = os.path.getsize(actual) / 1024
            print(c(f"  ✓ Saved → {actual} ({size_kb:.1f} KB)", GREEN))
        except Exception as e:
            print(c(f"  Save failed: {e}", RED))

    def _cmd_load(self, path: str):
        if not path:
            print(c("  Usage: /load <path>", YELLOW))
            return
        if not os.path.exists(path):
            print(c(f"  File not found: {path}", RED))
            return
        from agent import AgentSession
        from memory_socket import MemorySocket, AccessMode
        try:
            self.agent = AgentSession.resume(path)
            self.capsule_path = path
            n = len(self.agent.memory.graph.nodes)
            print(c(f"  ✓ Loaded '{path}' — {n} memories", GREEN))
        except Exception as e:
            print(c(f"  Load failed: {e}", RED))

    def _cmd_verbose(self):
        self.verbose = not self.verbose
        state = c("on", GREEN) if self.verbose else c("off", YELLOW)
        print(c(f"  Verbose ingestion: {state}", GREY))

    def _cmd_clear(self):
        self.agent.history = []
        print(c("  Conversation history cleared. Memories intact.", GREY))

    def _cmd_quit(self):
        print(c("  Saving…", GREY))
        try:
            actual = self.agent.save(self.capsule_path)
            size_kb = os.path.getsize(actual) / 1024
            print(c(f"  ✓ Capsule saved → {actual} ({size_kb:.1f} KB)", GREEN))
        except Exception as e:
            print(c(f"  Warning: could not save capsule: {e}", YELLOW))
        print(c("  Goodbye.", DIM))
        sys.exit(0)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="uMemory — plug-and-play agent memory chat CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python cli.py
              python cli.py --name myagent
              python cli.py --capsule session.umc
              python cli.py --capsule alex.umc --name alex
        """),
    )
    parser.add_argument(
        "--capsule", "-c",
        metavar="PATH",
        help="Path to a .umc capsule file to load",
        default=None,
    )
    parser.add_argument(
        "--name", "-n",
        metavar="NAME",
        help="Session/agent name (used as default save filename)",
        default="agent",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Show memory pipeline output (what gets extracted, stored, skipped)",
    )
    args = parser.parse_args()

    cli = MemoryCLI(capsule_path=args.capsule, name=args.name, debug=args.debug)
    cli.run()


if __name__ == "__main__":
    main()