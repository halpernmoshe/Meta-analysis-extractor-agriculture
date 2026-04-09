# Pipeline Improvements From Literature

## Purpose

This memo summarizes what the downloaded AI-evidence-synthesis literature suggests for improving the current pipeline.

The goal is not just to collect citations.

The goal is to extract **actionable design improvements** for:

- extraction
- adjudication
- normalization
- validation
- reporting
- governance

This note also connects those AI-era lessons to the older pre-LLM literature on how meta-analysis justifies truth-claims.

## Bottom Line

The strongest message from the literature is **not** “use more models everywhere.”

The strongest message is:

- build a cumulative validation framework
- make AI judgments auditable
- use targeted second-reviewer logic only for high-risk cases
- keep humans responsible without forcing them to read everything
- distinguish extraction error from alignment error from synthesis error
- explicitly quantify corpus bias and construct mismatch

That is highly consistent with what the current repo has already started to discover.

## Part I: Concrete Improvements Suggested By The AI-Evidence-Synthesis Literature

### 1. Add a formal AI-use disclosure and validation log

The 2025 position statement by Flemyng et al. makes a simple but important point:

- evidence synthesists remain responsible
- AI use should be transparently disclosed
- judgments suggested or made by AI should be fully reported

Practical implication for this pipeline:

For each run, automatically save a structured AI-use record including:

- model name and version
- prompt or prompt hash
- stage where AI was used
- purpose of AI use
- validation evidence available
- known limitations
- whether human review occurred

This should be generated automatically and stored with the outputs.

Why this matters:

- it improves auditability
- it reduces confusion across interrupted runs
- it makes the pipeline scientifically easier to defend

### 2. Make row-level AI decisions auditable

The current pipeline increasingly relies on semantic post-processing.

That means every keep/exclude/flag/swap decision should carry provenance.

Recommended per-row fields:

- model name/version
- prompt hash
- adjudication timestamp
- rationale text
- source evidence or quoted table/figure provenance
- whether the row was escalated or auto-resolved

This is especially important for Stage 6 and Stage 7.

Why this matters:

- rows become contestable and reviewable
- difficult disagreements can be examined after the fact
- paper-level conclusions become traceable to row-level decisions

### 3. Use second-reviewer logic only for high-risk rows

The strongest workflow idea from the collaborative LLM and automation literature is not blanket duplication.

The better idea is **risk-triggered duplication**.

Instead of re-reading every paper or every row twice, invoke a second model / second extraction pass only when risk is high.

Recommended triggers:

- source type is `figure`
- extreme effect size
- missing variance
- low-confidence extraction
- difficult comparator identity
- likely intervention confounding
- repeated-measures / non-independence risk
- possible treatment/control swap

This is a much better use of compute and review effort than universal double extraction.

### 4. Separate extraction error from alignment error

This is one of the most important lessons for the project.

The literature often evaluates “accuracy” as one number, but your system has already shown that this bundles together different failure modes.

Recommended explicit evaluation layers:

1. **Raw extraction quality**
   - Did the model read the values correctly from the paper?

2. **Alignment / matching quality**
   - Did the extracted row get matched to the right reference-standard row?

3. **Semantic adjudication quality**
   - Did the system correctly decide whether the row belongs in the synthesis?

4. **Synthesis-level agreement**
   - Does the final pooled effect match the published benchmark?

These should not be collapsed into one score.

### 5. Use explicit risk-based human oversight

The position statement does not imply that humans must reread every paper.

It implies that humans remain responsible for the synthesis and for how AI is used.

This suggests a practical policy:

- humans do not reread the whole corpus
- humans audit only predeclared edge cases

Recommended human-review triggers:

- figure-only extraction
- highly extreme effects
- outcome ambiguity
- intervention isolation ambiguity
- non-independence / repeated measures
- benchmark mismatch that diagnostics cannot explain

This preserves the core ambition of the project:

- minimal human reading of primary studies

while remaining scientifically defensible.

### 6. Track performance by variable type

The extraction literature repeatedly shows that some fields are much easier than others.

Performance should be tracked separately for:

- treatment/control means
- sample size
- variance type
- variance value
- categorical moderators
- outcome class
- setting class
- comparator identity

This would help identify where the real bottlenecks are instead of hiding them inside a single overall metric.

### 7. Add predeclared acceptance thresholds and stop-run rules

This is one of the most practical improvements available.

Before running a topic fully, define objective conditions that force review or rerun.

Examples:

- more than X% of rows are figure-derived
- more than X% of rows are low-confidence
- more than X% of rows lack variance
- more than X% of rows are flagged for estimand mismatch
- more than X% of rows require treatment/control swap
- spot-check disagreement exceeds threshold

This turns the pipeline into a governed instrument rather than a blind automation chain.

### 8. Quantify corpus bias explicitly

Several of the project’s hardest failures appear to come from differences between the accessible OA corpus and the benchmark corpus.

That means corpus composition should be treated as a formal diagnostic object.

For each topic, summarize:

- geography
- crop/species composition
- field vs pot vs greenhouse
- year range
- study duration
- comparator patterns
- intervention subtypes

Then compare that distribution to the benchmark’s explicit or implicit composition.

This would make benchmark disagreements much more interpretable.

## Part II: What Older Meta-Analysis Literature Suggests About “Truth”

Before LLMs, the meta-analysis literature already knew that meta-analysis does not find truth merely by pooling more papers.

Meta-analysis is informative only when:

- the estimand is clear
- heterogeneity is understood
- publication bias is considered
- comparator and outcome definitions are coherent
- study dependence is managed
- the protocol prevents opportunistic redefinition

In other words, older meta-analysis methodology already supports a key lesson that your pipeline is rediscovering:

**truth-like inference depends on disciplined construct definition, not just statistical aggregation.**

### Key pre-LLM lessons that matter here

1. **Heterogeneity is not noise to be ignored**
   - It often means the pooled estimate is mixing different constructs.

2. **Publication and availability bias change what the pooled effect means**
   - In your pipeline, OA bias is a concrete example of this problem.

3. **Protocol discipline matters**
   - Without a fixed benchmark spec and clear inclusion rules, replication becomes post hoc tuning.

4. **Dependence matters**
   - Repeated measures, multiple outcomes from the same paper, or overlapping controls can distort pooling.

5. **Construct validity matters more than convenience**
   - If the pipeline and benchmark estimate different things, statistical comparison is misleading.

This means the strongest pre-LLM methodological support for your current direction is not “meta-analysis finds truth automatically.”

It is:

- meta-analysis can support justified inference only when constructs, bias, and heterogeneity are carefully controlled

That maps directly onto why V2 needs:

- benchmark specs
- adjudication
- normalization
- diagnostics

## Part III: The Deeper Implication For This Project

The current project is moving from an **extraction-centric** paradigm to a **construct-aware synthesis** paradigm.

That is the correct move.

The extraction paper establishes that AI can read papers.

The pipeline work is establishing that reading papers is only one piece of the epistemic problem.

The real problem is:

- can the system identify which extracted rows actually belong in the target causal/statistical question?

That is a much more ambitious and interesting contribution.

## Part IV: Recommended Immediate Improvements To The Repo

### High Priority

1. Add an auto-generated AI-use disclosure file for every run.
2. Add row-level audit provenance for adjudication decisions.
3. Add retry / failure categorization that separates quota failures from real empty extractions.
4. Add stop-run thresholds before claiming topic completion.
5. Add corpus-bias diagnostics relative to benchmark composition.

### Medium Priority

6. Add risk-triggered second-reviewer extraction for high-risk rows only.
7. Add separate evaluation reports for extraction vs alignment vs adjudication vs synthesis.
8. Add variable-type-specific performance summaries.

### Lower Priority But Conceptually Important

9. Write a formal validation philosophy section for the next paper:
   - trust should come from multiple partially independent lines of evidence
   - not from one benchmark or one human rereading workflow

## Part V: What This Means For The Next Paper

The literature supports a stronger claim than:

- “LLMs can extract numbers from papers”

but a narrower claim than:

- “fully autonomous science is solved”

The strongest defensible claim is something like:

- autonomous evidence synthesis can be justified by a cumulative validation framework combining extraction accuracy, semantic adjudication, benchmark replication, internal consistency checks, and targeted human review of flagged cases

That is both methodologically and philosophically stronger than a simple extraction-accuracy paper.

## Key Sources In This Folder

### Core governance / best-practice source

- `pdfs/Flemyng_2025_AI_Evidence_Synthesis_Position_Statement.pdf`

### Strong workflow / automation source

- `pdfs/OttoSR_2025_Automation_of_Systematic_Reviews_with_LLMs.pdf`

### Supporting extraction / evaluation sources

- `pubmed_html/Gartlehner_2024_proof_of_concept.html`
- `pubmed_html/Gartlehner_2025_study_within_reviews.html`
- `pubmed_html/Jansen_2025_generative_AI_extraction.html`
- `pubmed_html/Kataoka_2026_GPT4o_o3_extraction.html`
- `pubmed_html/Khan_2025_collaborative_LLMs_living_reviews.html`
- `pubmed_html/Khan_2025_PMC_fulltext.html`
- `pubmed_html/Schmidt_Mathes_living_review_semiautomation.html`
- `pubmed_html/Gougherty_Clipp_2024_ecology_pubmed.html`
- `pubmed_html/ICASR_2024_summary.html`

## Final Judgment

The literature does not mainly tell this project to add more raw model power.

It tells the project to add:

- governance
- auditability
- risk-based escalation
- better decomposition of failure types
- explicit corpus/construct diagnostics

Those are exactly the improvements most likely to turn the current pipeline from a promising system into a scientifically defensible one.
