$proj = "C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor"

Set-Location "$proj\pipeline_replication"

$prompt = @"
You are restarting work on this project.

First reacquaint yourself with the whole project, including:
- the submitted paper in SUBMISSION_v23/SUBMISSION_CLEAN
- all current work under pipeline_replication/codex
- the current pipeline replication folders and topic outputs

Read codex/CLAUDE_HANDOFF.md first.

Then inspect the key codex notes, the replication pipeline, and the submitted manuscript materials.

Treat the existing topics as retrospective development/stress-test cases for Pipeline v2, not confirmatory results.
The universal downloader worked much better than previous downloaders and should be retained as a strength.

All new work in this phase should stay under pipeline_replication/codex unless there is a strong reason not to.

Work autonomously. Do not stop after giving a plan or high-level summary. After reacquainting yourself with the project, immediately start implementing Pipeline v2 work.

Use the already-downloaded papers from the already-done topics as the retrospective development bench for Pipeline v2.

Priorities:
1. Read codex/CLAUDE_HANDOFF.md and the key codex design notes.
2. Reacquaint yourself with the submitted paper and the current pipeline structure.
3. Design and implement universal Pipeline v2 improvements, especially:
   - stronger LLM-based post-extraction adjudication
   - canonical outcome / intervention / comparator / setting / estimand labels
   - non-independence handling
   - benchmark-spec-driven prospectively reusable logic
4. Test v2 against the already-downloaded old-topic corpora.
5. Keep producing concrete artifacts, scripts, notes, and outputs under codex.

Do not treat reruns on the old topics as confirmatory results. Use them only as retrospective development and stress testing.

Persist until blocked by a real technical limitation such as authentication failure, model unavailability, or a missing file. If blocked, leave the clearest possible status in codex and continue with any unblocked subtask.
"@

claude `
  --model opus `
  --dangerously-skip-permissions `
  --add-dir "$proj" `
  --add-dir "$proj\pipeline_replication\codex" `
  --add-dir "$proj\SUBMISSION_v23\SUBMISSION_CLEAN" `
  $prompt
