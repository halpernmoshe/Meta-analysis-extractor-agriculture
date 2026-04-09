#!/usr/bin/env python3
"""
Small driver for the multi-role full-context prototype.

Current capabilities:

1. validates that the six role files exist
2. reports which roles have substantive claims / constraints
3. can emit ready-to-run prompt files for each role
4. can invoke Claude Code live for selected roles
5. reruns the merger for that paper
6. writes a compact status JSON for downstream inspection
"""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
CODEX_DIR = ROOT / "pipeline_replication" / "codex"
PILOT_DIR = CODEX_DIR / "outputs" / "multi_role_pilot"
MERGER = CODEX_DIR / "merge_multi_role_pilot_outputs.py"
ROLE_NAMES = [
    "design_agent",
    "narrative_agent",
    "table_agent",
    "figure_agent",
    "benchmark_agent",
    "consistency_agent",
]

load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / ".env.kimi", override=False)

TEXT_ROLE_NAMES = [
    "design_agent",
    "narrative_agent",
    "table_agent",
    "benchmark_agent",
    "consistency_agent",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def role_payload(data: dict) -> dict:
    if "output_schema" in data and isinstance(data["output_schema"], dict):
        payload = dict(data["output_schema"])
        payload.setdefault("paper_id", data.get("paper_id"))
        payload.setdefault("role", data.get("role"))
        return payload
    return data


def render_role_prompt(role_file_data: dict) -> str:
    payload = role_payload(role_file_data)
    context = role_file_data.get("context", {})
    schema = payload
    prompt = role_file_data.get("prompt", "")
    parts = [
        f"Role: {role_file_data.get('role', '')}".strip(),
        "",
        "Task:",
        prompt,
        "",
        "Paper Context:",
        f"- paper_id: {role_file_data.get('paper_id', '')}",
        f"- pdf_paths: {json.dumps(context.get('pdf_paths', []), indent=2)}",
        f"- report_path: {context.get('report_path', '')}",
        "",
        "Report Excerpt:",
        context.get("report_excerpt", ""),
        "",
        "Output Requirements:",
        "Return strict JSON matching this schema shape. Preserve the role and paper_id.",
        json.dumps(schema, indent=2),
        "",
        "Instructions:",
        textwrap.dedent(
            """
            - Use the full paper context, not just one section.
            - Keep claims short and specific.
            - Prefer benchmark-relevant distinctions over generic summaries.
            - If the paper supports multiple nearby constructs, separate them.
            - If there is a contradiction or mismatch, state it explicitly in contradictions or notes.
            - Return only JSON.
            """
        ).strip(),
    ]
    return "\n".join(parts)


def find_claude_cli() -> str:
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path
    if sys.platform == "win32":
        candidates = [
            Path.home() / ".claude" / "local" / "claude.exe",
            Path.home() / "AppData" / "Local" / "Programs" / "claude" / "claude.exe",
            Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd",
            Path.home() / "AppData" / "Roaming" / "npm" / "claude",
            Path.home() / "AppData" / "Local" / "Programs" / "claude" / "claude.cmd",
            Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError("Claude Code CLI ('claude') not found in PATH.")


def extract_first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_claude_json_output(raw_output: str) -> dict | None:
    if not raw_output.strip():
        return None
    try:
        wrapper = json.loads(raw_output)
    except json.JSONDecodeError:
        wrapper = None

    wrapper_like = False
    if isinstance(wrapper, dict):
        wrapper_like = any(k in wrapper for k in ("result", "usage", "subtype", "is_error"))
        text_content = wrapper.get("result", wrapper.get("text", wrapper.get("content", "")))
        if isinstance(text_content, str):
            obj = extract_first_json_object(text_content)
            if obj is not None:
                return obj
    elif isinstance(wrapper, list):
        parts = []
        for block in wrapper:
            if isinstance(block, dict):
                parts.append(block.get("text", block.get("content", str(block))))
            elif isinstance(block, str):
                parts.append(block)
        obj = extract_first_json_object("\n".join(parts))
        if obj is not None:
            return obj

    if wrapper_like:
        return None
    return extract_first_json_object(raw_output)


def parse_openai_json_output(raw_output: str) -> dict | None:
    if not raw_output.strip():
        return None
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    return extract_first_json_object(raw_output)


def run_live_role(claude_cli: str, role_file: Path, timeout_seconds: int = 600) -> dict:
    raw = load_json(role_file)
    prompt = render_role_prompt(raw)
    attempts_dir = role_file.parent / "live_role_attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = attempts_dir / f"{role_file.stem}.attempt.json"
    started = time.time()
    try:
        result = subprocess.run(
            [
                claude_cli,
                "--print",
                "--output-format",
                "json",
                "--allowedTools",
                "Read",
                "--permission-mode",
                "bypassPermissions",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding="utf-8",
            errors="replace",
            input=prompt,
        )
    except subprocess.TimeoutExpired:
        payload = {
            "success": False,
            "error": f"Claude CLI timed out after {timeout_seconds}s",
            "duration_seconds": time.time() - started,
            "attempt_path": str(attempt_path),
        }
        attempt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    except Exception as e:
        payload = {
            "success": False,
            "error": f"Failed to run Claude CLI: {e}",
            "duration_seconds": time.time() - started,
            "attempt_path": str(attempt_path),
        }
        attempt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    duration = time.time() - started
    raw_output = result.stdout or ""
    stderr_output = result.stderr or ""
    if "You're out of extra usage" in raw_output or "You're out of extra usage" in stderr_output:
        payload = {
            "success": False,
            "error": "Claude Code extra usage exhausted",
            "duration_seconds": duration,
            "raw_output_preview": raw_output[:1000] or stderr_output[:1000],
            "attempt_path": str(attempt_path),
        }
        attempt_path.write_text(
            json.dumps(
                {
                    **payload,
                    "raw_output": raw_output,
                    "stderr_output": stderr_output,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return payload
    parsed = parse_claude_json_output(raw_output)
    if result.returncode != 0 and parsed is None:
        payload = {
            "success": False,
            "error": f"claude exited with code {result.returncode}: {stderr_output[:500]}",
            "duration_seconds": duration,
            "raw_output_preview": raw_output[:1000],
            "attempt_path": str(attempt_path),
        }
        attempt_path.write_text(
            json.dumps(
                {
                    **payload,
                    "raw_output": raw_output,
                    "stderr_output": stderr_output,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return payload
    if parsed is None:
        payload = {
            "success": False,
            "error": "Failed to parse JSON object from Claude output",
            "duration_seconds": duration,
            "raw_output_preview": raw_output[:1000],
            "attempt_path": str(attempt_path),
        }
        attempt_path.write_text(
            json.dumps(
                {
                    **payload,
                    "raw_output": raw_output,
                    "stderr_output": stderr_output,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return payload

    backup = role_file.with_suffix(".prelive.json")
    if not backup.exists():
        backup.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    raw["output_schema"] = parsed
    raw["live_metadata"] = {
        "runner": "run_multi_role_paper.py",
        "duration_seconds": duration,
        "claude_cli": claude_cli,
        "stderr_preview": stderr_output[:500],
    }
    role_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    attempt_path.write_text(
        json.dumps(
            {
                "success": True,
                "duration_seconds": duration,
                "claims": len((parsed.get("claims") or [])),
                "constraints": len((parsed.get("constraints") or [])),
                "contradictions": len((parsed.get("contradictions") or [])),
                "backup_path": str(backup),
                "attempt_path": str(attempt_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "success": True,
        "duration_seconds": duration,
        "claims": len((parsed.get("claims") or [])),
        "constraints": len((parsed.get("constraints") or [])),
        "contradictions": len((parsed.get("contradictions") or [])),
        "backup_path": str(backup),
        "attempt_path": str(attempt_path),
    }


def build_kimi_client(model_override: str | None = None) -> OpenAI:
    api_key = (
        os.environ.get("MOONSHOT_API_KEY")
        if "os" in globals()
        else None
    )
    if api_key is None:
        import os as _os
        api_key = _os.environ.get("MOONSHOT_API_KEY")
        base_url = _os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
        model = _os.environ.get("MOONSHOT_MODEL", "kimi-k2.5")
    else:
        import os as _os
        base_url = _os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
        model = _os.environ.get("MOONSHOT_MODEL", "kimi-k2.5")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY not found in environment.")
    client = OpenAI(api_key=api_key, base_url=base_url)
    client._codex_model = model_override or model  # lightweight attached metadata
    client._codex_base_url = base_url
    return client


def kimi_model_for_role(role_name: str) -> str:
    import os as _os
    text_model = _os.environ.get("MOONSHOT_TEXT_MODEL", "kimi-k2-thinking-turbo")
    vision_model = _os.environ.get("MOONSHOT_VISION_MODEL", _os.environ.get("MOONSHOT_MODEL", "kimi-k2.5"))
    if role_name == "figure_agent":
        return vision_model
    return text_model


def run_live_role_kimi(client: OpenAI, role_file: Path, timeout_seconds: int = 600, role_name: str = "") -> dict:
    raw = load_json(role_file)
    prompt = render_role_prompt(raw)
    attempts_dir = role_file.parent / "live_role_attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = attempts_dir / f"{role_file.stem}.attempt.json"
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=client._codex_model,
            max_tokens=8192,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "enabled"}},
            timeout=timeout_seconds,
        )
    except Exception as e:
        payload = {
            "success": False,
            "error": f"Failed to run Kimi API: {e}",
            "duration_seconds": time.time() - started,
            "attempt_path": str(attempt_path),
        }
        attempt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    duration = time.time() - started
    content = response.choices[0].message.content or ""
    parsed = parse_openai_json_output(content)
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    if parsed is None:
        payload = {
            "success": False,
            "error": "Failed to parse JSON object from Kimi output",
            "duration_seconds": duration,
            "raw_output_preview": content[:1000],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "attempt_path": str(attempt_path),
        }
        attempt_path.write_text(
            json.dumps(
                {
                    **payload,
                    "raw_output": content,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return payload

    backup = role_file.with_suffix(".prelive.json")
    if not backup.exists():
        backup.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    raw["output_schema"] = parsed
    raw["live_metadata"] = {
        "runner": "run_multi_role_paper.py",
        "provider": "kimi",
        "role_name": role_name or role_file.stem,
        "duration_seconds": duration,
        "base_url": client._codex_base_url,
        "model": client._codex_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    role_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    attempt_path.write_text(
        json.dumps(
            {
                "success": True,
                "provider": "kimi",
                "role_name": role_name or role_file.stem,
                "duration_seconds": duration,
                "claims": len((parsed.get("claims") or [])),
                "constraints": len((parsed.get("constraints") or [])),
                "contradictions": len((parsed.get("contradictions") or [])),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "backup_path": str(backup),
                "attempt_path": str(attempt_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "success": True,
        "provider": "kimi",
        "role_name": role_name or role_file.stem,
        "duration_seconds": duration,
        "claims": len((parsed.get("claims") or [])),
        "constraints": len((parsed.get("constraints") or [])),
        "contradictions": len((parsed.get("contradictions") or [])),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "backup_path": str(backup),
        "attempt_path": str(attempt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_id", help="Paper directory under outputs/multi_role_pilot")
    parser.add_argument("--emit-prompts", action="store_true", help="Write per-role prompt text files for live role runs")
    parser.add_argument("--run-roles", action="store_true", help="Invoke Claude Code and write live role outputs for the selected roles")
    parser.add_argument("--roles", type=str, default="", help="Comma-separated subset of roles to inspect/emit, e.g. design_agent,table_agent")
    parser.add_argument("--timeout-seconds", type=int, default=600, help="Timeout per live role call (default: 600)")
    parser.add_argument("--provider", choices=["claude", "kimi"], default="claude", help="Live role provider")
    args = parser.parse_args()

    paper_dir = PILOT_DIR / args.paper_id
    if not paper_dir.exists():
        print(f"Paper dir not found: {paper_dir}", file=sys.stderr)
        sys.exit(1)

    selected_roles = ROLE_NAMES
    if args.roles.strip():
        requested = [r.strip() for r in args.roles.split(",") if r.strip()]
        unknown = [r for r in requested if r not in ROLE_NAMES]
        if unknown:
            print(f"Unknown role(s): {', '.join(unknown)}", file=sys.stderr)
            sys.exit(1)
        selected_roles = requested

    role_status = []
    missing = []
    live_runs = []
    prompt_dir = paper_dir / "live_role_prompts"
    if args.emit_prompts:
        prompt_dir.mkdir(parents=True, exist_ok=True)
    claude_cli = ""
    kimi_clients = {}
    if args.run_roles:
        if args.provider == "claude":
            claude_cli = find_claude_cli()
        else:
            for role in selected_roles:
                model = kimi_model_for_role(role)
                if model not in kimi_clients:
                    kimi_clients[model] = build_kimi_client(model_override=model)
    for role in selected_roles:
        path = paper_dir / f"{role}.json"
        if not path.exists():
            missing.append(role)
            continue
        raw = load_json(path)
        payload = role_payload(raw)
        claims = payload.get("claims", []) or []
        constraints = payload.get("constraints", []) or []
        contradictions = payload.get("contradictions", []) or []
        prompt_path = None
        if args.emit_prompts:
            prompt_path = prompt_dir / f"{role}.prompt.txt"
            prompt_path.write_text(render_role_prompt(raw), encoding="utf-8")
        if args.run_roles:
            if args.provider == "claude":
                live_result = run_live_role(claude_cli, path, timeout_seconds=args.timeout_seconds)
            else:
                model = kimi_model_for_role(role)
                live_result = run_live_role_kimi(
                    kimi_clients[model],
                    path,
                    timeout_seconds=args.timeout_seconds,
                    role_name=role,
                )
            live_result["role"] = role
            live_runs.append(live_result)
            raw = load_json(path)
            payload = role_payload(raw)
            claims = payload.get("claims", []) or []
            constraints = payload.get("constraints", []) or []
            contradictions = payload.get("contradictions", []) or []
        role_status.append(
            {
                "role": role,
                "path": str(path),
                "n_claims": len(claims),
                "n_constraints": len(constraints),
                "n_contradictions": len(contradictions),
                "substantive": bool(claims or constraints or contradictions),
                "prompt_path": str(prompt_path) if prompt_path else "",
            }
        )

    subprocess.run(
        [sys.executable, str(MERGER), "--paper", args.paper_id],
        check=True,
    )

    merged_summary_path = paper_dir / "merged_summary.json"
    merged_summary = load_json(merged_summary_path) if merged_summary_path.exists() else {}

    status = {
        "paper_id": args.paper_id,
        "paper_dir": str(paper_dir),
        "selected_roles": selected_roles,
        "missing_roles": missing,
        "prompt_dir": str(prompt_dir) if args.emit_prompts else "",
        "live_runs": live_runs,
        "role_status": role_status,
        "merged_summary_path": str(merged_summary_path),
        "merged_summary": merged_summary.get("summary", {}),
    }

    out_path = paper_dir / "run_status.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print(out_path)


if __name__ == "__main__":
    main()
