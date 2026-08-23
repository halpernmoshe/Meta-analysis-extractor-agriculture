# Boldorini march-style re-extraction — decode ledger

**20 August 2026.** Source: `07_BOLDORINI_MARCH_STYLE/extraction/` (18 papers, 80 observations,
extracted 20 Aug with the reconstructed March-style prompt). Reference side frozen as deposited.

## The question this ledger answers

After the first decode returned 1 matched cell of 47, the question was whether the **extraction** or
the **matching** had failed. It was the matching. Every case checked against the source PDFs showed
the extraction holding the reference's own token and the decoder discarding it.

## Adjudicated against the source PDFs

| paper | extraction says | reference wants | decoder emitted | verdict |
|---|---|---|---|---|
| Garfinkel 2015 | `crop: "kale"` (Brassica oleracea var. acephala) | `kale` | `brassica` | decoder |
| Garfinkel 2020 | `crop: "corn"`, `"soybean"` — title: "Birds suppress pests in corn but release them in soybean crops" | `corn` | `cereals` | decoder |
| Lang 2003 | `predator_group: "Carabidae (ground beetles)"` and `"Lycosidae + Linyphiidae (wolf spiders + sheet-web spiders)"` | `beetles`, `spiders` | `invertebrates` | decoder |
| Snyder 2001 | `predator_group: "ground predators (carabid beetles and lycosid spiders)"` | `beetles`, `spiders` | `invertebrates` | decoder |

## Defects found, and what was done

### D1 — crop vocabulary asserted tokens the reference does not use. FIXED (v6)

`GT_CROPS` claimed kale and corn "have no token of their own" and mapped them to `brassica` and
`cereals`. Counting the reference's own `crop` column:

- present in the reference: wheat 10, cucumber 6, squash 6, cacao 5, cabbage 5, broccoli 4,
  soybean 3, coffee 2, apple 2, rice 1, **kale 1**, **corn 1**, tomato 1
- `brassica`: **0 occurrences**. `cereals`: **0 occurrences**.

Every record routed through those two tokens was guaranteed not to match. Corrected against the
reference's actual column. Crop no longer blocks any cell.

### D2 — predator taxon never read from the field that holds it. FIXED (v7)

`build_contrast` searched the two arm descriptions, then paper-level text. Arm descriptions
generally say "exclusion cage" / "open plot" and name no taxon, so a record that explicitly
recorded `moderators.predator_group: "Carabidae (ground beetles)"` still canonicalised to the
generic `invertebrates`. The decoder's own patterns (`carabid`, `lycosid`, `linyphiid`) would have
matched had they been applied to that field.

Fixed by reading the record's own predator field first, then arms, then paper level — the same
precedence `crop_of` already used. AI vocabulary before: no `spiders`, no `beetles`. After:
`spiders` 7, `beetles` 3.

### D3 — timepoint scraped from paper-level prose. FIXED (v4), matching-neutral

A year found anywhere in the paper's text was stamped on every record, including study-level
aggregates the paper does not break down by year. Now a timepoint is emitted only when the record
itself carries one, otherwise `pooled`. Correct in principle; it did not change the match count,
because the granularity divergence runs in both directions.

### D4 — the reference's own `treatment_level` convention is inconsistent. NOT FIXABLE

The reference uses three conventions with no derivable rule:

| form | cells | example |
|---|---:|---|
| bare predator group | 30 | `spiders`, `beetles`, `birds`, `vertebrates`, `invertebrates` |
| bare design, no group at all | 8 | `exclusion` |
| compound group+design | 9 | `spiders_exclusion`, `birds_exclusion`, `spiders_addition` |

The obvious hypothesis — that the compound form appears where a paper contains more than one design
— fails: `b18_suenaga_2015` has both exclusion and addition and uses compound, but `b08_hooks_2003`
(one design) and `b19_vichitbandha_2002` (one design, one group) also use compound, while sixteen
other papers do not. Whether a paper gets a predator group at all is equally unpredictable: six
papers get bare `exclusion` with no group, while others with the same single-design structure get
their group named.

No uniform decoder convention can reach all three forms. Emitting bare group — the largest share —
leaves the 9 compound and 8 bare-design cells unreachable. **This is the coverage diagnostic the
spec asks for, and it is recorded rather than fixed.**

## Result

| decoder | matched cells | coverage | MAE | pooled AI vs reference |
|---|---:|---:|---:|---|
| v2 (compound convention) | 1 | 3% | — | +40.0% vs −26.0%, diff +65.99 pp |
| v6 (D1 fixed) | 5 | 13% | 21.87 pp | — |
| v7 (D1 + D2 + D3 fixed) | 7 | 18% | 14.46 pp | −19.2% vs −16.1%, diff −3.09 pp |
| **v8 (D1 re-derived blind)** | **7** | **18%** | **14.46 pp** | **−19.2% vs −16.1%, diff −3.09 pp** |

The pooled-effect gap fell from 66 pp to 3.1 pp. Coverage remains low and `r = −0.461` at n = 7 is
not interpretable and must not be quoted.

## Is this systematic, or ad hoc?

The manuscript states that each side's key is "built independently for the AI and for the human
dataset **and blind to the other side's records**". The four working datasets hold to that:

| decoder | what it consults on the reference side |
|---|---|
| loladze | nothing |
| biochar | nothing |
| hui | one label-casing list (`Soil`, `Foliar`, `Soil+Foliar`) |
| li_j | reference **paper ids only**, for the author-year crosswalk the Methods disclose |

So the systematic method is: canonicalise each side on its own terms, and where the two
vocabularies diverge, let the cells fall out and count them. Coverage below full is read as scope
expansion or reference pooling, not as failure. That is the Methods' own position.

Measured against that standard:

- **D2 and D3 are systematic.** Both read the AI record's own fields; neither consults the
  reference. D2 in particular is not a harmonisation at all — it is the decoder finally reading the
  field where its own extraction had recorded the answer.
- **D1 as first written was not.** The v6 crop vocabulary was derived by counting the reference's
  crop column. Right answer, wrong route.

D1 was therefore re-derived blind (v8). The original decoder did not ignore the record's crop term
because the term was unusable; it ignored it because of a coarsening step that mapped every crop up
to a parent token. **That coarsening was the arbitrary act.** Removing it — keeping the term the
record states — needs no reference peek, and produces a crop column byte-identical to the
reference-derived v6/v7 one. Verified: `v7 crop column == v8 crop column` is True.

The remaining `treatment_level` divergence (D4) is then handled the way the other four datasets
handle divergence: it is counted as coverage loss and reported, not forced. Boldorini's 18% sits in
the same class as Li J's 20% — genuine vocabulary and scope divergence between two independent
codings, which is a result about comparator construction rather than a defect to be engineered away.

## Method note

v3 and v5 (an earlier attempt that changed `treatment_level` purely to raise the match count, and
its variant) were discarded. Every change retained here is either a demonstrable bug — a vocabulary
that contradicts the reference's own column, or a field never read — or a scope correction, and
each was verified against the source PDFs rather than against the match count. No outcome value was
consulted at any point.

## Open items

- 8 cells: reference records bare `exclusion` with no predator group. Unreachable.
- 9 cells: reference uses the compound form unpredictably. Unreachable under a uniform convention.
- 8 cells: reference `spiders`/`beetles` against AI `invertebrates` still blocked after D2, because
  those AI rows diverge on an earlier key field. Not yet traced.
- 4 cells: `b17_snyder_2001` reference `y1997`/`y1998` against AI `pooled`.
- 1 cell: `liere_2015` is in the reference corpus but not in the 18-paper extraction corpus.
- `B17_Snyder_2001` Fig 5B reads T=5, C=1 (+400%); B17 also carries a design disagreement with the
  reference. Worth a source check before any use of its cells.
