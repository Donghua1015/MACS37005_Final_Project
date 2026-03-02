#!/usr/bin/env python3
"""
Export a Claude Code session (.jsonl) to a readable Markdown file.

Usage:
    python export_chat.py                         # auto-detect newest session in current project
    python export_chat.py <session.jsonl>         # specific session file
    python export_chat.py <session.jsonl> -o out.md   # custom output path
    python export_chat.py --tools                 # also include tool calls/results
    python export_chat.py --all                   # export all sessions in current project
"""

import json
import re
import sys
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def strip_system_tags(text: str) -> str:
    """Remove IDE context tags injected by Claude Code."""
    text = re.sub(r'<ide_opened_file>.*?</ide_opened_file>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ide_selection>.*?</ide_selection>', '', text, flags=re.DOTALL)
    text = re.sub(r'<system-reminder>.*?</system-reminder>', '', text, flags=re.DOTALL)
    text = re.sub(r'<user-prompt-submit-hook>.*?</user-prompt-submit-hook>', '', text, flags=re.DOTALL)
    return text.strip()


def fmt_time(ts: str) -> str:
    """Format ISO timestamp to readable local time."""
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        local = dt.astimezone()
        return local.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return ts


def get_text(content_blocks: list) -> str:
    """Extract plain text from a content block list."""
    parts = []
    for b in content_blocks:
        if b.get('type') == 'text':
            parts.append(b['text'])
    return '\n'.join(parts)


def get_tool_calls(content_blocks: list) -> list:
    """Extract tool_use blocks."""
    return [b for b in content_blocks if b.get('type') == 'tool_use']


def get_tool_results(content_blocks: list) -> list:
    """Extract tool_result blocks."""
    return [b for b in content_blocks if b.get('type') == 'tool_result']


def truncate(s: str, max_len: int = 2000) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f'\n\n… *(truncated, {len(s) - max_len} more chars)*'


# ── core export ──────────────────────────────────────────────────────────────

def load_session(path: Path) -> list:
    msgs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return msgs


def build_thread(msgs: list) -> list:
    """
    Return messages in conversation order using parentUuid chain.
    Only main thread (isSidechain=False), only user/assistant types.
    """
    uuid_map = {}
    for m in msgs:
        if m.get('uuid') and m.get('type') in ('user', 'assistant'):
            uuid_map[m['uuid']] = m

    # Build children map
    children = {}
    roots = []
    for m in uuid_map.values():
        parent = m.get('parentUuid')
        if parent and parent in uuid_map:
            children.setdefault(parent, []).append(m)
        else:
            roots.append(m)

    # DFS traversal (stable order by timestamp)
    def dfs(node):
        yield node
        kids = sorted(children.get(node['uuid'], []), key=lambda x: x.get('timestamp', ''))
        for kid in kids:
            yield from dfs(kid)

    result = []
    roots_sorted = sorted(roots, key=lambda x: x.get('timestamp', ''))
    for root in roots_sorted:
        for node in dfs(root):
            if not node.get('isSidechain', False):
                result.append(node)

    return result


def export_to_md(session_path: Path, output_path: Path, include_tools: bool = False):
    msgs = load_session(session_path)
    thread = build_thread(msgs)

    session_id = None
    model = None
    start_time = None
    end_time = None
    tool_call_count = 0

    lines = []
    turn_num = 0

    for m in thread:
        mtype = m.get('type')
        content = m.get('message', {}).get('content', [])
        ts = m.get('timestamp', '')

        if start_time is None:
            start_time = ts
        end_time = ts

        if session_id is None:
            session_id = m.get('sessionId', '')

        # ── Human turn ───────────────────────────────────────────────────────
        if mtype == 'user':
            text_blocks = [b for b in content if b.get('type') == 'text']
            tool_results = get_tool_results(content)

            # Real user message (has text, not just tool results)
            if text_blocks and not tool_results:
                raw_text = get_text(text_blocks)
                clean = strip_system_tags(raw_text)
                if not clean:
                    continue
                turn_num += 1
                lines.append(f'\n---\n')
                lines.append(f'## 👤 User  <sup>{fmt_time(ts)}</sup>\n')
                lines.append(clean)
                lines.append('')

            # Tool results (only shown if --tools)
            elif tool_results and include_tools:
                for tr in tool_results:
                    tool_call_count += 1
                    result_content = tr.get('content', '')
                    if isinstance(result_content, list):
                        result_content = '\n'.join(
                            b.get('text', '') for b in result_content if b.get('type') == 'text'
                        )
                    lines.append(f'\n<details><summary>🔧 Tool result (id: {tr.get("tool_use_id","?")[:8]}…)</summary>\n')
                    lines.append(f'\n```\n{truncate(str(result_content))}\n```\n')
                    lines.append('</details>\n')

        # ── Assistant turn ────────────────────────────────────────────────────
        elif mtype == 'assistant':
            msg_obj = m.get('message', {})
            stop_reason = msg_obj.get('stop_reason')
            if model is None:
                model = msg_obj.get('model', '')

            text_blocks = [b for b in content if b.get('type') == 'text']
            tool_calls = get_tool_calls(content)

            # Final text response
            if text_blocks and stop_reason == 'end_turn':
                turn_num += 1
                lines.append(f'\n---\n')
                lines.append(f'## 🤖 Assistant  <sup>{fmt_time(ts)}</sup>\n')
                lines.append(get_text(text_blocks))
                lines.append('')

            # Tool calls (only shown if --tools)
            elif tool_calls and include_tools:
                for tc in tool_calls:
                    tool_call_count += 1
                    tool_name = tc.get('name', 'unknown')
                    tool_input = tc.get('input', {})
                    # Pretty-print input
                    if isinstance(tool_input, dict):
                        if 'command' in tool_input:
                            input_str = f"```bash\n{tool_input['command']}\n```"
                        elif 'file_path' in tool_input:
                            input_str = f"`{tool_input.get('file_path')}`"
                        else:
                            input_str = f"```json\n{truncate(json.dumps(tool_input, ensure_ascii=False, indent=2))}\n```"
                    else:
                        input_str = str(tool_input)

                    lines.append(f'\n<details><summary>🔧 Tool: <b>{tool_name}</b> (id: {tc.get("id","?")[:8]}…)</summary>\n')
                    lines.append(f'\n{input_str}\n')
                    lines.append('</details>\n')

    # ── Build header ─────────────────────────────────────────────────────────
    header = []
    header.append(f'# Claude Code — Chat Export\n')
    header.append(f'| Field | Value |')
    header.append(f'|---|---|')
    header.append(f'| Session ID | `{session_id}` |')
    header.append(f'| Model | `{model}` |')
    header.append(f'| Started | {fmt_time(start_time or "")} |')
    header.append(f'| Ended | {fmt_time(end_time or "")} |')
    header.append(f'| Turns | {turn_num} |')
    if include_tools:
        header.append(f'| Tool calls | {tool_call_count} |')
    header.append(f'| Source file | `{session_path.name}` |')
    header.append(f'\n> Exported on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by export_chat.py\n')

    full_md = '\n'.join(header) + '\n' + '\n'.join(lines)

    output_path.write_text(full_md, encoding='utf-8')
    print(f'Exported → {output_path}  ({turn_num} turns)')
    return output_path


# ── auto-detect session paths ─────────────────────────────────────────────────

def find_project_dir() -> Path | None:
    """Find the .claude/projects/<encoded-cwd> directory for the current working directory."""
    cwd = Path.cwd()
    # Encode the path the same way Claude Code does: replace separators with -
    encoded = str(cwd).replace('\\', '-').replace('/', '-').replace(':', '').replace(' ', '-')
    # Also try lowercased version
    claude_root = Path.home() / '.claude' / 'projects'
    if not claude_root.exists():
        return None
    # Find best match
    candidates = list(claude_root.iterdir())
    for c in candidates:
        if encoded.lower() in c.name.lower() or c.name.lower() in encoded.lower():
            return c
    # Fallback: most recently modified directory
    dirs = [c for c in candidates if c.is_dir()]
    if dirs:
        return max(dirs, key=lambda p: p.stat().st_mtime)
    return None


def find_sessions(project_dir: Path) -> list[Path]:
    return sorted(project_dir.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Export Claude Code session to Markdown.')
    parser.add_argument('session', nargs='?', help='Path to .jsonl session file (optional, auto-detects newest)')
    parser.add_argument('-o', '--output', help='Output .md file path')
    parser.add_argument('--tools', action='store_true', help='Include tool calls and results')
    parser.add_argument('--all', dest='all_sessions', action='store_true', help='Export all sessions in project dir')
    args = parser.parse_args()

    # Resolve session file(s)
    if args.session:
        session_files = [Path(args.session)]
    else:
        project_dir = find_project_dir()
        if project_dir is None:
            print('Could not find .claude/projects directory. Please pass a session file directly.')
            sys.exit(1)
        sessions = find_sessions(project_dir)
        if not sessions:
            print(f'No .jsonl files found in {project_dir}')
            sys.exit(1)
        if args.all_sessions:
            session_files = sessions
        else:
            session_files = [sessions[0]]
            print(f'Auto-detected newest session: {sessions[0].name}')

    # Export
    for sf in session_files:
        if args.output and len(session_files) == 1:
            out = Path(args.output)
        else:
            out = sf.with_suffix('.md')
        export_to_md(sf, out, include_tools=args.tools)


if __name__ == '__main__':
    main()
