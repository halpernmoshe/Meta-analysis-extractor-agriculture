# Tighter Post-Extraction Rules

This document defines a stricter prototype post-extraction layer that sits after the current `summary_validated.csv` outputs.

The goal is not to rewrite the preregistered analyses. The goal is to test whether tighter, benchmark-aware row filtering improves benchmark comparability while preserving transparency.

## Principles

1. Keep the main pipeline untouched.
2. Operate on current validated CSVs as input.
3. Make filtering rules explicit and topic-specific.
4. Prefer excluding clearly off-target rows over keeping ambiguous rows.
5. Treat these outputs as a diagnostic strict-pass, not a replacement for the preregistered primary results.

## Topic Rules

## `organic_yield_gap`

Primary target:
- harvested crop yield outcomes only

Exclude:
- quality traits
- concentration traits
- protein, mineral, flour, energy, ratio, hectolitre weight
- non-harvest product outcomes

Keep:
- grain yield
- fruit yield
- tuber yield
- total harvested yield
- equivalent grain yield if clearly yield-like

Reason:
- current validated data still contains non-yield outcomes that should not be in a primary yield-gap synthesis

## `notill_tillage`

Primary target:
- grain yield only

Exclude:
- straw yield
- biological yield
- forage / biomass proxies
- residual non-strict tillage rows if the description remains ambiguous

Keep:
- grain yield
- crop grain yield
- seed yield only if it is the crop yield endpoint

Reason:
- benchmark is about yield, and the current validated data still mixes grain with straw / biological outcomes

## `mycorrhiza_yield`

Primary target:
- crop yield and aboveground harvested productivity

Exclude:
- root biomass
- root dry weight
- belowground biomass
- photosynthetic efficiency / quantum traits
- nutrient uptake traits

Keep:
- grain yield
- fruit yield
- pod yield
- aboveground biomass / shoot biomass only when clearly productivity-related

Reason:
- this topic is mostly coherent already, but some physiological or belowground outcomes remain weakly aligned with the benchmark target

## `legume_rotation`

Primary target:
- subsequent crop yield after legume rotation vs control rotation / monoculture

Exclude:
- inoculation / PGPR / rhizobia / AMF manipulation rows
- pod-weight style component outcomes
- non-yield physiological traits

Keep:
- grain yield
- crop yield
- total dry matter only if it is explicitly the main crop productivity endpoint

Reason:
- some surviving rows look like biological inoculation studies rather than pure crop-rotation comparisons

## `biochar_crop_yield`

Primary target:
- field crop-yield comparisons compatible with the benchmark field-study design

Exclude:
- pot / greenhouse / chamber studies
- root biomass
- shoot dry weight and other non-harvest biomass endpoints
- highly artificial weight/weight dosing contexts when they are clearly pot-based

Keep:
- field grain yield
- field crop yield
- field harvested biomass only if clearly used as the crop-yield endpoint

Reason:
- the benchmark is field-focused; the current validated dataset still mixes field and pot contexts

## `intercropping_yield`

Primary target:
- system productivity / `LER` only

Exclude:
- individual crop yield rows from the primary replication comparison

Keep:
- `Land equivalent ratio`
- explicit system productivity equivalents

Reason:
- benchmark estimand is system productivity, not component crop yield

## Expected Effects Of Tightening

- `organic_yield_gap`: should remove residual quality leakage
- `notill_tillage`: should become a cleaner grain-yield synthesis
- `mycorrhiza_yield`: likely small change, but cleaner target alignment
- `legume_rotation`: should remove biological inoculation contamination
- `biochar_crop_yield`: should become more benchmark-comparable
- `intercropping_yield`: likely sparse but conceptually correct

## Important Interpretation Rule

These stricter outputs should be treated as:

- diagnostic strict-pass analyses
- benchmark-alignment sensitivity analyses

They should not silently replace the preregistered primary results.
