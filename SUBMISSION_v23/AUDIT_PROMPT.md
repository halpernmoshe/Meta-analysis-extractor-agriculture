# Submission Audit Prompt (v2)

Paste this into Codex or another auditor. The working directory is the SUBMISSION_v23 folder.

---

## Prompt

You are auditing a scientific manuscript submission package for internal consistency and submission readiness. The package is in the current directory. Do NOT modify any files -- only report findings.

### Files to review

- `PAPER_FINAL_v23.md` -- Main manuscript (Markdown source)
- `halpern_2026_v3.docx` -- Main manuscript (Word, current build)
- `halpern_2026_supplementary.docx` -- Supplementary materials (Word)
- `supplementary/SUPPLEMENTARY_MATERIALS.md` -- Supplementary source
- `supplementary/Table_S1_TOST_results.csv` through `Table_S4_agent_replication.csv`
- `figures/` -- 6 main figure PNGs (only 3 used in manuscript; see Notes item 10)
- `supplementary/Figure_S*.png` -- 3 supplementary figure PNGs
- `reproduction/` -- Self-contained reproduction package
- `README.md`, `COVER_LETTER.md`, `CODE_AVAILABILITY.md`

### Checks to perform

#### 1. Reference integrity (AMA numbered style)
- Confirm all 34 references are numbered sequentially (1-34) in the reference list.
- Confirm every superscript citation in the body text (e.g., ^1^) has a corresponding numbered reference.
- Confirm every numbered reference in the list is cited at least once in the body text.
- **Confirm references are in order of first appearance.** Walk through the manuscript from top to bottom, noting the first occurrence of each reference number. Ref 1 must appear before ref 2, ref 2 before ref 3, etc. Compound citations like ^14,17^ count as a first appearance for both 14 and 17 at that location.
- Confirm AMA formatting: Author LastName Initials, no periods between initials, journal names abbreviated and italicized, Year;Volume(Issue):Pages format.
- Flag any remaining APA-style citations (e.g., "(Author et al., 2023)") that were not converted.

#### 2. Table and figure cross-references
- List every table caption ([TABLE N: ...]) and figure caption ([FIGURE N: ...]) in the manuscript.
- Confirm each table/figure number is sequential (no gaps, no duplicates).
- Confirm every table and figure mentioned in running text (e.g., "Table 7", "Figure 2") has a corresponding caption.
- **Confirm every captioned table/figure has at least one mention in the running text** (not just in its own caption line, but in a separate prose sentence).
- For supplementary items (Table S1-S4, Figure S1-S3), confirm each is listed in the Supplementary Material section at the end of the manuscript.

#### 3. Numerical consistency
- For each dataset row in Table 4 (agreement metrics), verify that the same r, MAE, and N values appear consistently wherever that dataset is discussed in the text (abstract, results, discussion, conclusion).
- For Table 9 (run-to-run reproducibility), verify the Total row sums: 41+24+30 = 95 papers, 665+362+204 = 1,231 obs.
- Verify that aggregate effect differences mentioned in text (0.01-1.61 pp) are consistent with per-dataset values in the results section.
- Cross-check Table 5 (TOST results) against Table S1 -- the proportional 20% margin rows should match.
- Cross-check Table 9 against Table S4 CSV -- same datasets, same matched obs counts, same papers, same effect diffs.
- Verify the "1,149 observations from 136 papers" claim appears consistently in abstract, results, and conclusion.

#### 4. Supplementary consistency
- Confirm the supplementary document title matches the main manuscript title.
- Confirm all dataset labels use "Hui 2025" (not "Hui 2023") across all supplementary CSVs and markdown.
- Verify Table S4 data: Loladze should show 665 matched obs / 41 papers (not 1231/95 which are totals).
- Verify Table S2 tier summary (Excellent/Good/Fair/Poor counts and percentages) is internally consistent.

#### 5. DOCX content verification
- Extract text from `halpern_2026_v3.docx` and verify:
  - Title matches PAPER_FINAL_v23.md line 1.
  - **No literal caret marks (^) remain anywhere** -- not in body text, not in table cells, not in the reference list. All `^N^` markers should be rendered as superscript in the Word document.
  - Reference list contains numbered entries 1-34 in the new order (ref 17 = Gartlehner 2024, ref 20 = Anthropic, ref 25 = Loladze).
  - 3 embedded images are present.
  - Appendix A section is present.
- Extract text from `halpern_2026_supplementary.docx` and verify:
  - Title matches main manuscript title.
  - No "Hui 2023" instances remain.
  - 3 embedded supplementary images are present.
  - Table S4 shows Loladze = 665 obs / 41 papers.

#### 6. README and metadata
- Confirm README title matches the manuscript title.
- Confirm README lists the correct output filenames (halpern_2026_v3.docx).
- Confirm README says ~136 papers (not 137 or 138).
- Check COVER_LETTER.md: title should match manuscript, metrics should say 1,149 obs / 136 papers / proportional TOST / 0.01-1.61 pp.
- Check CODE_AVAILABILITY.md: title should match manuscript title.

#### 7. Reproduction package
- Run `python reproduce_everything.py` inside `reproduction/` and report:
  - How many checks pass vs fail.
  - Any numerical mismatches between reproduced values and manuscript claims.
  - Whether all 6 figure PNGs are regenerated.
  - Whether the docx build succeeds.
- If any check fails, report the exact mismatch (expected vs actual value).

#### 8. Writing quality flags
- Flag any remaining LLM-isms: "cornerstone", "noteworthy", "furthermore", "moreover", "notably", "underscores", "highlights the", "it is worth noting", "myriad", "plethora", "utilize", "facilitate", "delve", "elucidate", "paradigm" (when not referring to actual experimental paradigms), "landscape", "holistic", "nuanced", "multifaceted", "pivotal", "paramount".
- Flag any em-dashes (---) remaining in the text (en-dashes -- for ranges are fine).
- Flag any sentences that start with "It is important to note that" or similar throat-clearing.
- Flag duplicate/near-duplicate sentences across sections.

#### 9. Statistical consistency deep-check
- For each dataset in Table 4: verify ICC(3,1) value matches or is very close to the r value.
- Verify Cohen's d values in Section 3.2.1 are all < 0.20 as claimed.
- Verify the "5.5x" source-type accuracy claim: Table 8 shows Table median = 0.57 pp, Figure median = 3.12 pp. Confirm 3.12/0.57 = 5.47 which rounds to 5.5x.
- Verify "81% had zero extraction error" claim for Hui 2025 (Section 3.1) is plausible given N=319 obs.
- Check that direction agreement percentages in Table 4 are consistent with text claims.
- Verify Section 3.1 aggregate effect differences match Table 4 claims (Hui = 0.12, Li 2022 = 0.15, etc.).

#### 10. Orphaned or misplaced content
- Check for any text after the last supplementary figure caption -- there should be nothing.
- Check for any TODO, FIXME, XXX, HACK, or placeholder markers anywhere in the manuscript, cover letter, or code availability statement.
- Check for any lines that look like authoring notes (e.g., "[Note: ...]").
- Verify the Declarations section contains all required subsections: Competing Interests, Funding, Data Availability, CRediT, AI Tools disclosure.

#### 11. Cross-document title consistency
- The following files must all contain the exact same paper title: PAPER_FINAL_v23.md (line 1), COVER_LETTER.md, CODE_AVAILABILITY.md, README.md, supplementary/SUPPLEMENTARY_MATERIALS.md (supplementary title).
- Confirm: "Breaking the Extraction Bottleneck: A Single AI Agent Achieves Statistical Equivalence with Human-Extracted Meta-Analysis Data Across Five Agricultural Datasets"
- Flag ANY file where the title differs.

#### 12. Reference number spot-check
After confirming the sequential order in check 1, do a spot-check of 5 specific references to confirm the renumbering is internally consistent:
- Ref 15 in the reference list should be Gartlehner et al. 2024 (proof-of-concept study, *Res Synth Methods*). In-text, it should appear as ^14,15^ in Section 1.3 (cited alongside Gartlehner 2025 which is ref 14).
- Ref 16 should be Poser et al. 2026 (*Front Artif Intell*). First cited in Section 1.3 as ^16^.
- Ref 17 should be Khan et al. 2025 (*J Am Med Inform Assoc*). First cited in Section 1.3 as ^17^.
- Ref 20 should be Anthropic (Claude API pricing). First cited in Section 2.1.4.
- Ref 25 should be Loladze 2014 (*eLife*). First cited in Section 2.2.
- Ref 34 should be Flemyng et al. / Cochrane position statement. Cited in Sections 4.9 and 4.11.

### Output format

For each check category, report:
- PASS: [brief confirmation]
- FAIL: [exact location, expected value, actual value]
- WARN: [potential issue that may or may not need fixing]

Summarize with a final verdict: READY / NOT READY (with blockers listed).

---

## Notes on decisions made in this revision

The following changes were made to the submission package. An auditor may flag these as differences from earlier versions; they are intentional:

1. **References converted from APA author-date to AMA numbered style.** 34 references numbered by order of first appearance. All `(Author et al., YYYY)` citations replaced with superscript numbers.

2. **References renumbered for first-appearance order (two rounds).** Round 1: Gartlehner 2024 (originally ref 32), Khraisha 2024 (ref 33), and Li L. 2025 (ref 34) were cited in Section 1.3 but numbered after refs 17-31. These were renumbered: 32->17, 33->18, 34->19, and old refs 17-31 shifted to 20-34. Round 2: On line 58, the compound citation ^14,17^ (Gartlehner 2025 + 2024) appeared before ^15^ (Poser) and ^16^ (Khan). This was fixed by rotating: old 17->15, old 15->16, old 16->17. Final order on line 58: ^13^, ^14,15^, ^16^, ^17^, ^18,19^. All in-text citations and the reference list have been updated.

3. **Growth environment details removed from Section 1.2.** The sentence "Key experimental systems include free-air CO2 enrichment (FACE) facilities, open-top chambers (OTC), and controlled-environment growth chambers" was removed per author request -- it was oddly specific for an introduction about agricultural data challenges in general. FACE/OTC/growth chamber terminology remains in the Loladze 2014 dataset description (Section 2.2) and Discussion (Section 4.4) where it is contextually appropriate.

4. **Supplementary title matched to main manuscript.** Changed from "...Equivalence with Human Coders Across Five Independent Meta-Analysis Datasets" to "...Statistical Equivalence with Human-Extracted Meta-Analysis Data Across Five Agricultural Datasets".

5. **"Hui 2023" standardized to "Hui 2025"** across all 4 supplementary CSVs and the supplementary markdown. The main paper already used "Hui 2025" consistently. The underlying data folder is still named `hui2023_full_35` internally (not user-facing).

6. **Table S4 corrected.** The Loladze row previously showed 1,231 obs / 95 papers (which are cross-dataset totals). Corrected to 665 / 41 per main paper Table 9. Li 2022 row populated with 204 / 30. Hui 2025 row populated with 362 / 24 and effect diff 6.31 pp. Row order now matches Table 9 (Loladze, Hui, Li).

7. **README updated** to list correct filenames and paper count (~136 papers).

8. **LLM-isms removed.** "cornerstone" -> "central to" (Section 1.1). "noteworthy" -> "stand out" (Section 4.7). Em-dashes in affiliation line replaced with commas.

9. **build_docx.py updated** with superscript rendering for `^N^` citation markers in both body text AND table cells, and proper handling of numbered references in the AMA reference list (preserving visible numbers with hanging indent).

10. **Figures 4-6 are intentionally not in the main manuscript.** The paper has 3 main figures (scatter, per-paper MAE, Bland-Altman) and 3 supplementary figures (S1-S3). The `figures/` folder contains 6 PNGs because `generate_figures.py` produces all of them, but only figures 1-3 are placed in the manuscript via `[FIGURE N: ...]` captions. The extra PNGs (architecture, source-type accuracy, aggregate effects) are available as supplementary assets but are not referenced in the main text. This is a deliberate editorial decision, not a build error.

11. **Running-text cross-references added.** Every table (Tables 1-10) and figure (Figures 1-3) now has at least one mention in running prose, in addition to its caption line. Previous version only had explicit running-text mentions for Table 7.

12. **COVER_LETTER.md and CODE_AVAILABILITY.md updated.** Titles matched to current manuscript. Cover letter metrics updated to 1,149 obs / 136 papers / proportional TOST / 0.01-1.61 pp. Cover letter Key Contributions updated to reflect actual paper claims (removed stale "ground-truth-free validation" bullet, added run-to-run reproducibility).
