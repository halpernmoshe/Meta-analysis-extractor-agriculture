# Boldorini 2024 corpus — extraction

You are extracting quantitative data from primary research papers so that the result can be compared
against an independently compiled dataset. Read each source PDF directly. Return one JSON object per
paper, containing one record per observation.

Work paper by paper. Do not summarise across papers, and do not stop early: every paper in the
corpus gets a file, including any paper from which you could extract nothing, which gets a file
saying so.

---

## Scope

- **Population** — field or semi-field crop systems in which natural enemies of crop pests are
  present or manipulated.
- **Intervention** — manipulation of predator access: exclusion (cages, netting, exclosures) or
  addition (releases, augmentation).
- **Comparison** — the unmanipulated condition, that is natural or ambient predator levels.
- **Outcome** — crop yield. Also record arthropod pest abundance and crop damage where reported, but
  label them distinctly in `outcome`; never record them as yield.

## The treatment and control convention — read this twice

- `treatment_mean` is the arm in which predator access was **manipulated**: the excluded arm in an
  exclusion experiment, the augmented arm in an addition experiment.
- `control_mean` is the arm at **natural predator levels** — open plots, uncaged, no release.

This is the most common error in this corpus. An exclusion cage keeps predators **out**, so the
caged arm is the treatment and the open arm is the control — even though the cage may look like the
"experimental" condition and the open plot like the "untouched" one. Getting this backwards inverts
every effect in the paper at once.

In `treatment_description` and `control_description`, state exactly which physical arm each value
came from, in the paper's own words, so the assignment can be checked afterwards.

**Never infer the control by picking the lowest value, the first column, or the leftmost bar.** The
control is identified from what the paper says the arm *is*, never from what its number *is*. If you
cannot tell which arm is which, that observation is unresolved — say so and move on.

## Expected direction

Where predators suppress pests, excluding them is expected to **reduce** yield and to **increase**
pest abundance and damage. A result in the opposite direction may well be real — report it — but
re-check the treatment and control assignment before recording it, and note in `notes` that you
checked.

---

## Output schema

One JSON object per paper:

```json
{
  "paper_id": "B01_Ali_2018",
  "title": "...", "authors": "...", "year": 2018, "doi": "...", "journal": "...",
  "crop": "wheat", "species": "Triticum aestivum",
  "experiment_type": "field exclusion", "location": "Langfang, Hebei, China",
  "design": "...", "n_replicates": 6,
  "notes": "...",
  "observations": [ { "...one object per observation..." } ]
}
```

Each observation:

| field | content |
|---|---|
| `outcome` | what was measured: `crop yield`, `pest abundance`, `crop damage` |
| `tissue` | measured part or basis where relevant, e.g. grain, fruit, whole plant |
| `treatment_mean` | mean in the predator-manipulated arm |
| `control_mean` | mean at natural predator levels |
| `effect_pct` | `100 × (treatment_mean − control_mean) / control_mean` |
| `n` | independent replicates per arm — not subsamples, plants, or repeated measures |
| `unit` | exactly as the paper reports it |
| `variance_value` | as reported |
| `variance_type` | `SD`, `SE`, `CI`, `LSD`, `CV`, or `null` |
| `data_source` | precise location: table number, figure and panel, or page for text |
| `treatment_description` | the physical arm, in the paper's words |
| `control_description` | the comparator arm, in the paper's words |
| `moderators` | `predator_group`, `treatment_type` (`exclusion` / `addition`), `crop`, `study_year`, `country`, and any other condition distinguishing this observation |
| `figure_read` | required if and only if `data_source` is a figure — see the figure protocol |

## Rules

- Extract every eligible comparison the paper reports, not a sample.
- Never compute a mean the paper does not report. Do not average across arms to manufacture a
  contrast the authors did not present.
- If a paper reports several predator groups, several crops, several years, or several designs,
  record each as its own observation. Do not collapse them into a single contrast.
- If `n` is ambiguous between replicates and subsamples, record what the paper says and flag it in
  `notes`.
- If a value cannot be supported from the source, omit the observation and say why in `notes`. Do
  not guess.

---

## Figure protocol — mandatory for every figure-sourced value

Much of this corpus reports its yield results only in figures. The method is always the same:
**look at the graph and understand it first; only then digitise and measure it.** Understanding is a
separate step, with its own written output, and it comes first. A number read off a figure whose
structure you have not established is not an extraction, it is a guess.

**The governing rule: recover what is visually present, never infer what is not.** If a bar top, an
axis tick, or an error-bar cap is not separable by eye at high resolution, record a status code, not
a number. A blank with a reason is a correct result. A plausible guess is a wrong one — and worse
than a blank, because nothing downstream can tell it apart from a real read.

A figure-sourced value that does not carry a completed `figure_read` record is not valid and must
not be emitted.

### Stage A — Find the figure, and understand it

First locate the figure. Do not trust the page number printed in the paper; those are journal pages
and these PDFs do not start at page 1. Find the figure by searching the PDF text for its caption, and
confirm by reading the caption you find. If you cannot find it, the observation is
`not_on_referenced_page` — do not read a different figure instead.

Render the page at high resolution — 300 to 600 dpi, higher for dense or multi-panel plots — and
look at it.

Then, before reading any value, write down:

- how many panels there are, and what distinguishes them;
- how many bars or points per group, and what each one represents;
- **which element is the predator-manipulated arm and which is the natural-predator control, and how
  you can tell** — legend colour, hatching, position, axis label;
- the y-axis label, its unit, and every tick value with its spacing, stated explicitly
  ("ticks at 0, 500, 1000, 1500, 2000; spacing 500; unit kg/ha");
- whether the axis is linear or logarithmic;
- what the caption says the error bars represent.

This goes in `figure_read.structure`. **If you cannot complete this list, you do not read the
figure** — emit the observation with a status code instead.

### Stage B — Calibrate, and check the calibration

Pick two ticks whose values you can read, and note their pixel positions. Then predict a **third**
tick from that two-point calibration, and compare it against where the third tick actually sits. If
the prediction is off by more than about 3%, the calibration is wrong: re-read the ticks, or record
`axis_unclear` and leave the value blank. Record the check as a percentage.

This is the cheapest available guard against a misread axis, and it is not optional.

For a log axis, interpolate in log space and set `log_axis: true`.

### Stage C — Measure, then measure again

Read the bar top or point centre, and convert:

    linear:  value = vmin + (pos − pos_min) × (vmax − vmin) / (pos_max − pos_min)
    log:     value = 10 ^ ( log10(vmin) + (pos − pos_min) × (log10(vmax) − log10(vmin)) / (pos_max − pos_min) )

Then **go back to the figure and read it again from scratch** — re-locate the bar, re-read the axis,
re-derive the value. Record both readings.

- If they agree within 5%, report the second and set `reconciled: true`.
- If they differ by more than 5%, read a third time and take the two that agree. If none agree,
  record `reconciliation_failed` and leave the value blank.

State the position in words as well as in numbers, so the read can be checked: "bar top sits about
60% of the way between the 1000 and 1500 ticks, so about 1300".

Where a value is printed on the bar or point, use the printed number and say so — a printed value is
always preferred to a measured one.

### Stage D — Error bars

If a cap is clearly visible, read its position, convert it, and record
`variance_value = |value(cap) − value(mean)|`. Set `variance_type` to whatever the **caption** says
it is; do not assume. If no cap is readable, `variance_type: null`. Error-bar readings reconcile at
a looser 15% tolerance than means.

### Stage E — Check before committing

- Is the value inside the plotted axis range? A value outside it is a misread, not a finding.
- Is it plausible for this crop and this unit?
- Does the direction match what the arm assignment implies? If your read says the excluded arm
  out-yielded the open arm, **re-check which bar is which before recording it.** A treatment/control
  swap on a figure inverts every effect in the paper simultaneously, and it is the single most
  damaging error available here. If the direction still holds after re-checking, report it, and say
  in `notes` that you checked.

### The `figure_read` record

```json
"figure_read": {
  "figure_ref": "Fig 3b", "pdf_page": 6, "located_by": "caption search",
  "structure": "two panels by year; per group two bars, open = ambient predators, hatched = exclusion cage; y-axis grain yield kg/ha, ticks 0/500/1000/1500/2000, spacing 500, linear; caption states error bars are SE",
  "axis_min": 0, "axis_max": 2000, "axis_pixel_min": 812, "axis_pixel_max": 141,
  "log_axis": false, "calibration_check_pct": 1.2,
  "reading_1": 1305, "reading_2": 1290, "reconciled": true,
  "point_pixel": 375, "errorbar_pixel": 341,
  "status": "digitized",
  "evidence_note": "Fig 3b p6; hatched bar, group 'wheat 2016'; top ~60% between 1000 and 1500 ticks"
}
```

`status` is one of: `digitized`, `axis_unclear`, `too_crowded`, `bar_not_identifiable`,
`errorbar_not_identifiable`, `not_on_referenced_page`, `reconciliation_failed`.

Anything other than `digitized` means `treatment_mean` and `control_mean` stay null for that
observation. **Emit the observation anyway, with its status.** A recorded failure to read is data; a
silently dropped row is not.

---

## Corpus

Eighteen PDFs, all in:

    <folder holding the eighteen source PDFs>

    B01_Ali_2018_wheat_beetle.pdf              B10_Lang_2003_wheat_multiple.pdf
    B02_Bisseleua_2017_cacao_invert.pdf        B11_Libran-Embid_2017_coffee_vertebrate.pdf
    B03_Borkhataria_2012_rice_bird.pdf         B13_Maas_2013_cacao_bird.pdf
    B04_Classen_2014_coffee_vertebrate.pdf     B14_Martin_2013_brassica_bird.pdf
    B05_Garfinkel_2015_brassica_bird.pdf       B15_Mols_2002_apple_bird.pdf
    B06_Garfinkel_2020_cereal_bird.pdf         B16_Saunders_2016_apple_vertebrate.pdf
    B07_Gras_2016_cacao_bird.pdf               B17_Snyder_2001_generalist_predators_trophic_cascades.pdf
    B08_Hooks_2003_broccoli_spider.pdf         B18_Suenaga_2015_cabbage_spider.pdf
    B09_Ismoilov_2020_tomato_lacewing.pdf      B19_Vichitbandha_2002_soybean_invert.pdf

(There is no B12; the numbering is not contiguous.)

## Output

One file per paper, named `<paper_id>_agent.json`, written to:

    <output folder>

When all eighteen are written, produce `COMPLETENESS.md` in the same folder, reporting per paper:
the number of observations emitted, **the number carrying a non-null `treatment_mean`**, the number
sourced from tables versus from figures, and the count of each figure `status` code. Report those
counts separately — a manifest can be perfectly consistent about an incomplete extraction, so record
counts alone do not show whether the extraction succeeded.

## What this run must not read

Any earlier extraction of this corpus, the published Boldorini et al. 2024 supplementary dataset,
any key table, any matching output, or any result from this project. Read only the eighteen source
PDFs listed above.
