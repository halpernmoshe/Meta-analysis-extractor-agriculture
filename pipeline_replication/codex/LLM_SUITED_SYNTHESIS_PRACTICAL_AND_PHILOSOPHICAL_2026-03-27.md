# LLM-Suited Synthesis: Practical And Philosophical Notes

This note summarizes the line of thought developed in the current discussion and extends it into a proposal for a broader kind of evidence synthesis better matched to LLM-based systems.

## 1. The Immediate Practical Problem

The original goal looked straightforward:
- extract quantitative observations from papers
- compare them to published datasets or meta-analyses
- decide whether the system is right

But in practice the failure modes were not mostly simple numeric hallucinations.

They were more often:
- concentration vs content
- foliar vs grain
- EC-only vs EC+EO
- final harvest vs intermediate harvest
- pooled benchmark value vs subgroup-specific extracted value
- figure-only targets vs table-only extractions
- structurally valid extraction but wrong benchmark alignment

This means that the core difficulty is not only:
- "Can the LLM read numbers?"

It is:
- "Can the system tell whether the extracted number belongs to the target construct?"

That is a different and deeper problem.

## 2. Why One-Channel Extraction Is Not Enough

A paper does not contain only one kind of evidence.

It contains:
- abstract-level claims
- results-text claims
- methods / design constraints
- tables
- figures and captions
- discussion / conclusion statements
- implicit constraints about what comparisons are allowed

A numeric extractor sees only one evidential slice of the paper, even if it reads the whole PDF.

That is why some wrong outputs can still be coherent:
- they fit the story of the paper
- they fit a nearby table row
- they even fit the direction of the conclusion
- but they still do not match the exact target estimand

So the right question is not just:
- "Did we extract a number?"

But:
- "Does the extracted claim cohere with the whole evidential structure of the paper?"

## 3. Practical Architecture: Full-Context, Multi-Role Reading

The best practical architecture is probably not:
- one giant monolithic extractor

and not:
- many context-starved tiny agents

The better architecture is:
- multiple agents
- all reading the full paper
- but with different epistemic roles

For example:

### Design agent
Reads the full paper and extracts:
- intervention/comparator structure
- valid treatment arms
- timepoints
- tissues
- units
- factorial structure
- what comparisons are scientifically allowed

### Narrative agent
Reads the full paper and extracts:
- abstract/result/conclusion claims
- direction of effect
- significance statements
- the paper's own verbal interpretation

### Table agent
Reads the full paper but focuses on:
- tabular numeric evidence
- means, variance, n
- explicit treatment/control rows

### Figure agent
Reads the full paper but focuses on:
- figure-only targets
- captions
- graphical trends
- whether the benchmark target exists only in figures

### Benchmark-comparability agent
Reads the full paper and a benchmark spec and asks:
- what exact construct does the benchmark appear to use?
- which paper rows fit that construct?
- which rows are nearby but non-equivalent?

### Consistency agent
Takes the outputs of the others and checks:
- contradictions
- unsupported comparisons
- wrong-arm or wrong-tissue usage
- narrative-vs-table conflict
- table-vs-design conflict

This is not just redundancy. It is structured plurality.

Each agent sees the whole paper, but each one has a different task, different schema, and different bias.

## 4. The Core Philosophical Shift

The deep problem is epistemological.

There is no perfect gold standard:
- humans are fallible
- benchmarks are fallible
- published meta-analyses are fallible
- LLMs are fallible

So the real question becomes:
- what counts as enough justification to trust a synthesis system?

The strongest answer developed so far is:

> Trust should come from convergence across multiple partially independent lines of evidence, not from any single allegedly infallible authority.

That is a philosophy-of-science position, not just an engineering trick.

It is close to:
- triangulation
- robustness reasoning
- construct validity
- measurement invariance
- distributed epistemology

This matters because many failures are not "false facts" in the simplest sense.
They are:
- construct drift
- measurement non-equivalence
- benchmark mismatch
- forced alignment

So the project naturally moves from:
- extraction validation

toward:
- epistemology of autonomous evidence synthesis

## 5. Why Meta-Analysis Is Not The Only Target

Traditional meta-analysis is built for:
- one defined outcome class
- a relatively fixed estimand
- numeric comparability across studies
- pooled effect estimation

That is powerful, but also restrictive.

Many papers contain much richer evidence than what a standard meta-analysis keeps:
- multiple tissues
- multiple timepoints
- multiple treatment arms
- figures without clean tables
- narrative design information
- qualitative explanations for why some numbers are not comparable

An LLM can read all of that. Traditional meta-analysis largely cannot represent it.

This suggests that LLM-based systems may be suited not only to automating existing synthesis forms, but to creating a new synthesis form.

## 6. A Possible New Form: Consilience Synthesis

One candidate name is:
- `Consilience synthesis`

The key idea:
- do not reduce a paper immediately to one pooled effect row
- instead represent the paper as a structured bundle of evidence

Each claim or topic would be summarized across dimensions such as:
- numeric grounding
- cross-model agreement
- within-paper support
- construct drift
- benchmark comparability
- rescue potential
- structural risk

This is not just narrative review.
It is not just meta-analysis either.

It is a structured evidence-integration system where:
- quantitative results still matter
- but so do design constraints and contradictions

In this model, the system might output something like:

### Claim
Elevated CO2 reduces grain Zn in wheat.

### Evidence profile
- numeric support: strong
- cross-model agreement: high
- within-paper support: high
- construct drift risk: low
- benchmark comparability: high
- rescue needed: none

or

### Claim
Elevated CO2 reduces Ca in this rice study.

### Evidence profile
- numeric support: moderate
- cross-model agreement: high
- within-paper support: moderate
- construct drift risk: high
- issue: grain GT matched against leaf values
- rescue potential: high if restricted to correct tissue / figure extraction

This gives a richer scientific object than a single effect size.

## 7. Other Possible Names For This New Synthesis Form

Depending on tone, this could be described as:
- Consilience synthesis
- Constraint-aware synthesis
- Construct-aware synthesis
- Multi-channel evidence synthesis
- Epistemic profile synthesis
- Comparative evidence synthesis

My current preference is:
- `construct-aware consilience synthesis`

because it emphasizes both:
- comparability of constructs
- convergence across evidence channels

## 8. What Makes This Better Suited To LLMs

LLMs are unusually good at:
- integrating many local clues
- comparing multiple representations of the same paper
- reasoning about design constraints
- translating between numeric and narrative evidence
- spotting near-matches that are not exact matches

They are not uniquely good at:
- pure numerical pooling
- exact statistical estimation

So the best division of labor is likely:

### LLMs do:
- paper reading
- construct specification
- cross-channel comparison
- contradiction finding
- benchmark alignment
- evidence profiling

### Classical code/statistics do:
- pooling
- uncertainty calculations
- bias diagnostics
- sensitivity analysis
- final effect estimation where appropriate

That suggests a synthesis pipeline that is not merely "LLM automates the old thing", but:
- LLMs create a richer, structured representation of study evidence
- classical statistics operate on the subset that is truly comparable

## 9. A Practical Research Program

This implies a research program with several stages.

### Stage 1. Extraction validation
Already underway / partially complete.

### Stage 2. Construct-drift diagnostics
Now underway.

### Stage 3. Consilience profiles
Represent claims using multiple evidence dimensions.

### Stage 4. Counterfactual rescue
Estimate whether a claim becomes valid under the correct arm/tissue/timepoint restriction.

### Stage 5. Hybrid synthesis
Combine:
- pooled quantitative evidence where possible
- structured consilience evidence where pooling is invalid or incomplete

### Stage 6. New publication format
The paper would not merely say:
- "the pooled estimate is X"

It would also say:
- how stable the claim is
- whether it is benchmark-comparable
- what drift risks exist
- whether the evidence converges across modalities

## 10. Philosophical Payoff

This solves part of the epistemic problem without pretending to eliminate uncertainty.

Instead of:
- one benchmark
- one extractor
- one effect size

you get:
- multiple readers
- multiple evidence channels
- structured contradictions
- explicit comparability checks
- transparency about where synthesis is secure and where it is not

That is a more honest model of science.

Science rarely works by one perfect measurement.
It works by constrained convergence among imperfect measurements.

This project may be heading toward a computational version of that principle.

## 11. Strongest Short Form Of The Idea

The most compact way to say it is:

> LLMs may be less important as automatic data extractors than as engines for construct-aware, multi-channel evidence integration.

And the corresponding methodological proposal is:

> The right successor to purely numeric AI meta-analysis may be a form of construct-aware consilience synthesis that integrates quantitative, narrative, design, and graphical evidence before statistical pooling.

## 12. Best Next Step

The best concrete next step is to prototype a full-context multi-role paper reader.

For a small set of papers:
- run a design agent
- run a narrative agent
- run a table agent
- run a figure agent
- run a benchmark-comparability agent
- compare outputs and record contradictions

Then test whether that architecture distinguishes:
- clean claims
- alignment problems
- coverage problems
- uncertain claims

better than the current single-layer extraction logic.
