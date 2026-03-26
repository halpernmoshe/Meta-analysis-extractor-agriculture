# Independent Audit Report
**Date**: 2026-03-26
**Auditor role**: Independent reviewer with no prior involvement in the project
**Files reviewed**: CLAUDE_HANDOFF.md, STATUS_LOG.md, PIPELINE_V2_ARCHITECTURE.md, humic_acid_yield/benchmark_spec.md, SPOT_CHECK_REPORT_2026-03-26.md, BENCHMARK_ALIGNED_ANALYSIS_2026-03-26.md, PIPELINE_V2_FROZEN_2026-03-26.md

---

## SECTION 1: What Is This Project? (Plain English)

This project is building a fully automated system — called the "pipeline" — that can replicate the results of published scientific reviews in agriculture without any human reading papers. The goal is ambitious: you give the system a topic (e.g., "does humic acid increase crop yield?"), it searches databases, downloads open-access research papers, reads them, extracts the relevant numbers, and produces a pooled statistical estimate comparable to what a team of scientists would produce after months of manual literature review. The system uses AI language models (LLMs) to do the reading and judgment, and the quality check is whether the pipeline's estimate matches that of a published meta-analysis that serves as a reference benchmark. If the pipeline can reliably reproduce published benchmarks across multiple topics, the system has demonstrated that it can function as an autonomous scientific synthesis engine — a significant result in the field of AI-assisted research.

---

## SECTION 2: What Has Been Built So Far?

### Core Infrastructure (Proven, Working)
- A universal PDF downloader that retrieves open-access papers from multiple sources (Unpaywall, OpenAlex, PubMed Central, publisher sites). This was identified as the best-performing component from earlier development and is explicitly retained in v2.
- A multi-model data extraction engine that uses two or more LLMs in parallel and takes a consensus reading of quantitative results from tables and text in scientific papers.
- A post-extraction statistical engine that converts different variance types (SE, SD, LSD, CI, CV) and computes log response ratios for meta-analytic pooling using DerSimonian-Laird random effects.

### Version 1 Learning Work (Complete)
- Full pipeline runs on 6 topics: legume rotation, mycorrhiza (fungi), organic farming yield gap, no-till tillage, biochar, and intercropping.
- A diagnostic analysis showing where V1 failed and why, covering all 6 topics.
- A keyword-based post-extraction adjudicator that applies rule-based filters to decide which extracted rows enter the final synthesis.
- Universal effector normalization: LLM-based labeling of crop type, study setting, climate zone, and soil type for every extracted row.
- Resynthesis comparison scripts that show before-and-after effects of each filtering step.

### Version 2 Design and Preparation (Complete as of 2026-03-25/26)
- A full 9-stage Pipeline V2 architecture specification, now frozen in `PIPELINE_V2_FROZEN_2026-03-26.md`.
- A candidate topic scoring exercise covering 18 potential topics, scored on 8 criteria, with a final 6-topic set selected for V2.
- Topic configuration files (JSON) for all 6 V2 topics.
- Benchmark specification documents for all 6 V2 topics — structured descriptions of what each reference meta-analysis measured, how, and what counts as a valid comparison.
- A written-but-not-yet-run LLM semantic adjudicator script (`adjudicate_llm_universal.py`) that replaces keyword-based filtering with AI-based judgment.
- Two analytical reports produced today (2026-03-26): a spot-check of keyword adjudication quality across 4 topics, and a benchmark-aligned subset analysis across all 6 topics.
- A preregistration-ready frozen architecture document.

---

## SECTION 3: What Was Done Today?

Today's work consisted of six distinct analytical steps. In plain English:

**Step 1: Spot-Check Report on Keyword Adjudication**
The system audited its own filtering decisions from V1. It asked: "When the pipeline excluded a row of data, was that the right call?" The report graded each of four topics (GOOD, ADEQUATE, POOR, POOR) and identified which failure modes keyword-based filtering cannot solve — specifically, that keywords cannot reliably distinguish legitimate yield outcomes from non-yield ones when authors use non-standard language, and cannot verify whether the "control" arm in a study is actually an untreated control or just a different treatment variant.

**Step 2: Benchmark-Aligned Subset Analysis**
The system tested whether restricting the analysis to rows that most closely match the benchmark paper's design criteria improves the estimate. The honest finding: this approach works in only one of six topics (biochar). For three topics it makes no difference, and for two it makes things worse. The analysis concluded that benchmark-alignment filtering is a useful diagnostic tool but a poor correction mechanism, and that the real fixes are upstream (in extraction configuration and intervention taxonomy).

**Step 3: Pipeline V2 Architecture Freeze**
The full 9-stage architecture was locked into a document marked FROZEN. This means the design rules — what happens at each stage, what the schemas look like, what the success criteria are — cannot be changed without creating a formal deviation log entry. This is essential for research integrity: it prevents the researchers from unconsciously adjusting the design to fit the results after the results are known.

**Step 4: Benchmark Specification Documents**
For each of the 6 V2 topics, a structured document was created that translates the reference benchmark paper's methods into explicit rules: what counts as the right intervention, what counts as the right comparator, what the outcome hierarchy is, and where the known ambiguity traps are. For the humic acid topic, for example, the spec spells out that compost and biochar are not valid HA interventions, and that biostimulant products only qualify if humic acid is the primary active ingredient. These specs allow the LLM adjudicator to evaluate rows against a precise standard rather than a vague topic label.

**Step 5: LLM Adjudicator Script**
A script was written that replaces the keyword-based adjudicator with a full LLM-based semantic judgment layer. Each extracted row is evaluated by Claude against the topic config, with a structured output specifying whether the row's intervention, comparator, outcome, and estimand match the benchmark's definition. The script exists and has been tested in dry-run mode; it is blocked from running on real data because both API keys (Anthropic and Google) are currently expired or out of credit.

**Step 6: Preregistration Preparation**
The topic set (6 topics), success criteria (direction agreement on 5/6 topics and CI overlap on 3/6 as primary thresholds), and full architecture were frozen. The project is now in a state where it can formally preregister: commit publicly to what it will measure, how it will measure it, and what result would count as success or failure, all before running the analysis.

---

## SECTION 4: Does It Make Sense?

### Is the Approach Scientifically Sound?

Yes, in its design philosophy. The three-layer architecture (broad extraction, deterministic QC, LLM semantic adjudication) is well-motivated by the failure modes documented in V1. The separation between "LLM for semantic judgment, code for math" is a principled and defensible division of labor. Using published meta-analyses as external reference benchmarks rather than as ground truth is epistemically correct — the project explicitly states that disagreement with a benchmark does not mean the pipeline is wrong, only that it diverged, and it builds a taxonomy for classifying why.

The preregistration approach is scientifically essential. Without it, any agreement between the pipeline and a benchmark could be dismissed as the product of iterative tuning. The freeze of the architecture before any V2 results are seen is the right move and is correctly executed here.

The benchmark specification documents are a genuine methodological contribution. Most automated extraction systems extract data against a vague topic label. These specs translate benchmark methods into explicit, machine-readable inclusion criteria. This is the right way to operationalize the comparison.

### Are the 6 Steps Logically Ordered?

Yes. The sequence is:
1. Audit old system (spot-check) — establish what needs fixing
2. Test whether a naive fix works (benchmark-aligned subset) — establish it does not
3. Freeze the new design — prevent post-hoc adjustment
4. Create the benchmark comparison targets — give the design something to aim at
5. Build the adjudicator — the core V2 improvement
6. Preregister — lock the evaluation

This is the correct order. Freezing before building the adjudicator would have been premature if the benchmark specs were not ready. Building the adjudicator before auditing V1 would have missed important failure modes. The sequence is defensible.

### Is Anything Premature, Redundant, or Missing?

**Premature**: Nothing appears premature. The architecture freeze is happening at the right moment — after enough diagnostic work to know what V2 should include, and before any V2 results have been generated.

**Redundant**: The benchmark-aligned subset analysis is arguably redundant with the earlier Codex effector normalization work (which tested similar filters through a different mechanism). However, the 2026-03-26 version is more systematic and produces cleaner conclusions, so the redundancy is productive rather than wasteful.

**Missing (important)**: The LLM adjudicator is written but cannot run. This is the most important missing piece. Until API keys are refreshed, V2 cannot execute its central innovation. Everything else — architecture, specs, config, scripts — is ready. The blocker is purely operational.

**Missing (secondary)**: There is no dress rehearsal result in the files reviewed. The PIPELINE_V2_FROZEN document says "dress rehearsal completed" in its version history, but no dress rehearsal output report was found in the codex folder. If a dress rehearsal was run, its results should be documented before preregistration, because surprises in a dress rehearsal sometimes require architecture changes that cannot be made after freezing.

---

## SECTION 5: Key Risks and Concerns

### Risk 1: API Keys Are Expired (Blocker)
Both the Anthropic and Google API keys are currently out of credit or expired. The LLM adjudicator — the central V2 innovation — cannot run. The entire V2 evaluation is blocked until this is resolved. This is a purely operational fix (add credit / generate new key), but it is the most urgent item on the list.

### Risk 2: The Architecture Was Frozen Without a Full Dress Rehearsal
The frozen architecture document says a dress rehearsal was completed, but no dress rehearsal results file was found in the reviewed documents. If the dress rehearsal revealed any issues that required architecture changes, and those changes were not logged as deviations, the integrity of the frozen document is undermined. This should be verified before any preregistration is submitted.

### Risk 3: The Humic Acid Pilot Topic Has Not Been Run End-to-End
The humic acid topic was selected as the V2 pilot specifically because it is clean, well-scoped, and never tested. However, there is no evidence in the reviewed files that the full 9-stage pipeline has been run on it. The benchmark spec is complete, the config is written, and the adjudicator is ready — but the actual execution has not happened. If the pilot reveals extraction problems (e.g., HA studies often conflate fulvic acid and humic acid products, the benchmark spec explicitly notes this ambiguity), the architecture may need adjustment — but it is now frozen.

### Risk 4: The LLM Adjudicator Has Only Been Dry-Run Tested
The script `adjudicate_llm_universal.py` was tested in dry-run mode, meaning no actual API calls were made. The logic of the script is untested against real data. For a system whose central innovation is LLM-based semantic judgment, this is a meaningful gap. The first live run on a V2 topic may reveal prompt failures, schema parsing errors, or unexpected model behavior that the dry run cannot anticipate.

### Risk 5: Two V1 Topics Have Structural Failures That Persist After All Fixes
The notill_tillage topic produces a wrong-direction result, and the intercropping_yield topic has a fundamental estimand mismatch (the pipeline computes per-component crop yield; the benchmark computes system productivity). Both of these are carried forward into the V2 topic set if the existing extracted data is reused. If the V2 evaluation is intended to be a clean forward-looking test, carrying structurally broken V1 data into it would contaminate the results. The evaluation design needs to be explicit about whether V2 reuses V1 extraction outputs or re-extracts from scratch.

### Risk 6: The Success Criteria Are Ambitious Relative to V1 Performance
V1 achieved correct direction in 4/6 topics (67%). The V2 primary success threshold is 5/6 (83%) direction agreement. This is achievable if the LLM adjudicator resolves the intervention-definition mismatch issues identified in V1, but it is not guaranteed. The two hardest V1 topics (notill, intercropping) may remain structurally difficult regardless of adjudication quality.

### Risk 7: Benchmark Papers Have Not All Been Verified for Availability
The benchmark specs name specific papers (e.g., Ma, Cheng & Zhang 2024 for humic acid; Ainsworth & Long 2021 for CO2). It is not confirmed in the reviewed files that the exact benchmark numbers (pooled effect, k, CI) have been extracted from these papers and recorded. The preregistration should include the exact benchmark values being targeted, and those values should be locked before any V2 analysis is run.

---

## SECTION 6: What the User Needs to Decide

### Decision 1: Refresh API Keys Now or After Preregistration?
The LLM adjudicator cannot run without active API keys. If the intent is to run a live dress rehearsal on the humic acid pilot topic before preregistering, keys must be refreshed first. If the intent is to preregister now and run V2 afterward, the keys can be refreshed after submission. However, running a live test before preregistering is strongly recommended because it may reveal issues that require architecture changes — which are no longer allowed after freezing.

### Decision 2: Was the Dress Rehearsal Completed, and Where Are the Results?
The frozen architecture document says "dress rehearsal completed" in its version history but no output report was found. The user should confirm: (a) was a dress rehearsal actually run, (b) what did it produce, and (c) are there any changes needed as a result? If the dress rehearsal was not run, the architecture should not yet be treated as frozen.

### Decision 3: Do the V2 Topics Reuse V1 Extracted Data or Re-Extract?
For the 4 topics carried forward from V1 (legume_rotation, elevated_co2, cover_crop, and possibly others), does the V2 evaluation start from a fresh search and extraction, or does it start from the already-downloaded and already-extracted V1 data? This decision affects whether V2 is truly a prospective evaluation. For research integrity, a clean re-run from search onwards is preferable. But for cost and time reasons, reusing V1 downloads with V2 adjudication is a reasonable alternative — as long as it is stated explicitly and registered.

### Decision 4: Should the Intercropping and No-Till Topics Be Replaced or Fixed?
These two topics have structural problems that are deeper than adjudication can fix. Intercropping needs a different estimand (LER instead of component yield). No-till needs a redefined intervention taxonomy. The question is whether the user wants to: (a) keep them in V2 with explicit re-extraction using corrected configs; (b) replace them with two cleaner topics from the scored candidate list; or (c) keep them and flag them in the preregistration as known-hard topics where structural failure is anticipated. Each option is defensible, but it must be chosen before freezing the topic set.

### Decision 5: Where Will V2 Be Preregistered, and When?
The architecture and success criteria are ready for preregistration. The practical questions are: on which platform (OSF, PROSPERO, AsPredicted, or a GitHub commit timestamp), in what format, and by what date. This does not require any additional technical work — it is an administrative decision about where and when to make the commitment public.

---

## Overall Assessment

The project is methodologically sound, well-documented, and in a mature state. The V1 development phase honestly catalogued its own failures rather than cherry-picking successes, which is the right scientific posture. The V2 architecture addresses the root causes of V1 failures in principled ways. The decision to preregister before running V2 results is the correct call.

The project is not yet ready to run V2 because the API keys are expired and the live adjudicator has not been tested on real data. The most important next action is to refresh API keys, run the humic acid pilot end-to-end with the LLM adjudicator, verify the results are reasonable, and then finalize and submit the preregistration. Everything needed to do that is already in place.

The single biggest scientific risk is that the architecture freeze may have happened before a genuine dress rehearsal on live data. If the live pilot reveals issues that require structural changes, those changes will need to be logged as deviations — which is procedurally possible but weakens the preregistration's credibility. Running the pilot before formal submission of the preregistration would eliminate this risk.
