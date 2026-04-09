# Live Role Runner Bridge

## What Was Added

`run_multi_role_paper.py` now supports direct live role execution through Claude Code CLI:

- `--run-roles`
- `--roles design_agent,benchmark_agent,...`
- `--timeout-seconds N`
- existing `--emit-prompts` path remains intact

The runner now:

1. discovers the local Claude CLI using Windows-aware lookup
2. renders a full-context per-role prompt from the scaffold JSON
3. invokes Claude Code with:
   - `--print`
   - `--output-format json`
   - `--allowedTools Read`
   - `--permission-mode bypassPermissions`
4. parses a returned JSON object
5. writes the parsed result back into the role scaffold as `output_schema`
6. preserves a one-time backup as `*.prelive.json`
7. records each attempt under `live_role_attempts/`

## Important Safety Fix

The first live run exposed a real failure mode:

- Claude returned a quota wrapper:
  - `You're out of extra usage`
- the original parser mistakenly treated the wrapper itself as a successful payload

This is now fixed.

Current behavior:

- quota / wrapper failures are classified as failed live runs
- the original role JSON is left untouched
- the failed wrapper is saved to `live_role_attempts/<role>.attempt.json`
- the failure is also recorded in `run_status.json`

## Verified Test

Tested on:

- `outputs/multi_role_pilot/019_Baxter_1994`
- role subset: `design_agent`

Observed result:

- live Claude invocation works mechanically
- current blocker is usage exhaustion, not runner logic
- role file remained intact after failure
- attempt record was written successfully

Relevant files:

- `outputs/multi_role_pilot/019_Baxter_1994/run_status.json`
- `outputs/multi_role_pilot/019_Baxter_1994/live_role_attempts/design_agent.attempt.json`
- `outputs/multi_role_pilot/019_Baxter_1994/design_agent.prelive.json`

## Practical Next Step

When Claude usage is available again, run a narrow live test first:

```powershell
python codex/run_multi_role_paper.py 019_Baxter_1994 --run-roles --roles design_agent,benchmark_agent,consistency_agent --timeout-seconds 600
```

Then inspect:

- updated role JSONs
- `merged_claims.csv`
- `merged_summary.json`
- per-role attempt records

If that succeeds, the next clean expansion is:

- `015_Pleijel_2009` as clean-support anchor
- `035_Oksanen_2005` as arm-mismatch case
- `026_Seneweera_1997` as figure-only / tissue-mismatch case
