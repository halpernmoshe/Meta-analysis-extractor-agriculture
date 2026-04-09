# Creative Ideas From Literature

This note pulls together literature that is not just about "LLMs for systematic reviews", but about how science justifies trust when no single procedure is perfect. The goal is to generate new design ideas for the convergence / extraction-risk framework.

## Core Literature Signals

### 1. Concordance helps, but is not enough

- Khan et al. report that concordant responses from collaborative LLM extraction outperform either model alone, and discordant cases should be cross-critiqued rather than trusted automatically.
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/39836495/
- This supports the current use of cross-model support as a positive signal.
- It does **not** support treating agreement as truth by itself.

### 2. Responsible AI in evidence synthesis requires transparency, justification, and human responsibility

- Cochrane, Campbell, JBI, and CEE's joint statement emphasizes transparent reporting, accountability, and preserving methodological rigor when AI is used in evidence synthesis.
  - Cochrane summary: https://www.cochrane.org/about-us/news/setting-standards-responsible-ai-use-evidence-synthesis
- This supports the move toward explicit risk buckets and audit trails rather than hidden heuristics.

### 3. Automation is feasible, but quality is the constraint

- Górska and Tacconelli describe a framework toward autonomous living meta-analyses and argue that screening and, to a large extent, extraction can be automated while maintaining strict review logic.
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/39176757/
- This supports trying to formalize a validation stack rather than abandoning autonomy.

### 4. Meta-analysis has always been about bias control plus comparability, not just pooling

- The James Lind Library history of meta-analysis emphasizes that systematic review and meta-analysis became distinct because reducing bias in study assembly, appraisal, and comparability is separate from statistical pooling.
  - James Lind Library: https://www.jameslindlibrary.org/articles/a-historical-perspective-on-meta-analysis-dealing-quantitatively-with-varying-study-results/
- This supports the current distinction between:
  - extraction quality
  - alignment / estimand matching
  - synthesis

### 5. Robustness / triangulation literature is directly relevant

- The philosophy-of-science literature on robustness and triangulation argues that trust comes from stability across partly independent routes, not from one infallible method.
  - Evidential diversity / triangulation discussion and references: https://www.cambridge.org/core/journals/philosophy-of-science/article/evidential-diversity-and-the-triangulation-of-phenomena/53FA073FD9560A4F07230E7C4D8E19C7
  - Robustness overview: https://link.springer.com/article/10.1007/s13194-025-00673-1
- This supports the project's deeper framing: the target is cumulative validation, not single-source proof.

### 6. Construct validity and measurement invariance suggest a new analogy

- Cronbach and Meehl's construct validity idea is that a measure is validated by a nomological network, not a single comparison.
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/13245896/
- Measurement invariance work asks whether the same construct is actually being measured across groups before comparing results.
  - PubMed example: https://pubmed.ncbi.nlm.nih.gov/27266799/
- This is highly relevant because many pipeline failures are effectively failures of measurement invariance:
  - the benchmark measured one construct
  - the pipeline extracted a related but not equivalent construct

### 7. Sensor-fusion anomaly detection has a useful structural analogy

- Sensor-fusion work shows that anomalous readings are easier to diagnose when you compare multiple nearby, partially redundant sensors rather than trusting one stream.
  - Example: https://www.mdpi.com/2224-2708/14/2/34
- This is analogous to using:
  - abstract
  - text
  - tables
  - figures
  - cross-model extractions
  - reruns
  as redundant evidence channels.

## Most Creative Design Ideas

### Idea 1. Replace "hallucination detection" with "measurement invariance testing"

Current framing:
- Is this row hallucinated or real?

More powerful framing:
- Is the pipeline and the benchmark measuring the same latent construct?

Implications:
- Build explicit tests for construct drift:
  - concentration vs content
  - foliar vs grain
  - EC-only vs EC+EO
  - pooled vs subgroup-specific
  - final harvest vs intermediate harvest
- Treat these as invariance failures, not just extraction failures.

Why this is promising:
- It matches the actual failure modes already seen in the corpus.
- It is a stronger conceptual bridge to philosophy of measurement and psychometrics.

### Idea 2. Build a "nomological network" for each extracted claim

Instead of only asking whether a row matches another row, ask whether it fits the paper's larger network of evidence:
- abstract direction
- results text direction
- table values
- figure captions
- conclusion language
- known agronomic priors
- cross-model agreement
- rerun stability

This turns each claim into a small construct-validity problem.

Why this is promising:
- It directly addresses coherent hallucinations.
- A hallucinated number may fit one table, but not the paper's whole network.

### Idea 3. Separate "sensor anomaly" from "real event" the way sensor-fusion papers do

Sensor anomaly detection distinguishes:
- faulty sensor
- genuine environmental event

The analogous distinction here is:
- extraction artifact
- genuine but different estimand

This suggests a formal three-way anomaly model:
- extraction error
- benchmark/estimand mismatch
- true scientific discordance

That is much better than a binary right/wrong label.

### Idea 4. Use reliability-weighted evidence fusion instead of flat agreement counts

Not all evidence channels are equally trustworthy.

Potential weighting scheme:
- table numeric cells: highest
- direct results text with numbers: high
- figure captions: medium
- figure digitization: medium-low
- abstract/conclusion narrative only: low
- same-model rerun agreement: lower independence than cross-model agreement

This could be implemented as a Dempster-Shafer style or Bayesian evidence-combination layer.

Why this is promising:
- It gives a formal home to the intuition that some convergence is weak and some convergence is strong.

### Idea 5. Introduce "consilience profiles" instead of a single risk score

Rather than one scalar hallucination-risk number, store a profile:
- numeric grounding
- cross-model concordance
- within-paper support
- construct invariance
- benchmark comparability
- report-level structural risk

Then classify papers/claims by profile shape, not just total score.

Why this is promising:
- It preserves diagnosis instead of collapsing everything to one number.
- It fits the project's 4-way policy and could naturally expand it.

### Idea 6. Add a "counterfactual rescue" analysis

For each bad paper, ask:
- If I remove figure-only targets, does the paper become clean?
- If I restrict to the benchmark's intended arm, does it become clean?
- If I restrict to the right tissue or timepoint, does it become clean?

This is already implicit in several reports. Make it explicit.

Why this is promising:
- It distinguishes pipeline failure from target-definition failure.
- It creates a causal-style diagnosis rather than just an error label.

### Idea 7. Use meta-analysis publication-bias ideas as a template for extraction-bias diagnostics

Publication-bias methods do not observe truth directly; they detect systematic distortions.

Analogous extraction-bias diagnostics:
- direction asymmetry by source type
- overrepresentation of easy table outcomes
- missingness concentrated in figure-only outcomes
- stronger disagreement in factorial designs
- bias toward narrative-consistent but numerically weak claims

Why this is promising:
- It moves the project from single-row truth checking toward corpus-level bias detection.

### Idea 8. Treat benchmark replication like external validity, not truth

The history of meta-analysis suggests that:
- study assembly
- appraisal
- comparability
are all part of the inferential problem.

So benchmark replication should be treated as:
- an external calibration test
not
- ground truth itself

Why this is promising:
- It resolves a major epistemic pressure point in the project.
- It aligns the paper with a stronger philosophy-of-science story.

## Practical Next Steps Suggested By This Literature

1. Add explicit claim-level "construct drift" flags:
   - concentration_vs_content
   - tissue_mismatch
   - arm_mismatch
   - timepoint_mismatch
   - pooled_vs_subgroup_mismatch

2. Convert the within-paper layer from a bag of keywords into a weighted evidence-channel model.

3. Build a "consilience profile" table rather than only a single rule score.

4. Add a corpus-level extraction-bias dashboard:
   - figure-only miss rate
   - factorial-paper error rate
   - wrong-tissue forced-match rate
   - narrative-only support rate

5. Write the philosophical framing around cumulative validation, construct validity, and invariance rather than "LLM proved right."
