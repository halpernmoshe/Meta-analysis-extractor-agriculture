#!/usr/bin/env python3
"""
resynthesize_all.py — Re-run synthesis on PICO-validated data for all three topics.

Reads summary_validated.csv (output of pico_validate.py) and runs the existing
synthesis scripts modified to use the validated data.

Usage:
    python resynthesize_all.py
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent


# ── Shared functions ─────────────────────────────────────────────────────────

def compute_lnRR(t_mean, c_mean):
    try:
        t, c = float(t_mean), float(c_mean)
        if t > 0 and c > 0:
            return math.log(t / c)
    except (TypeError, ValueError):
        pass
    return None


def lnRR_to_pct(lnRR):
    return (math.exp(lnRR) - 1) * 100


def compute_variance_lnRR(sd_t, sd_c, n_t, n_c, mean_t, mean_c):
    try:
        vals = [float(x) for x in (sd_t, sd_c, n_t, n_c, mean_t, mean_c)]
        if any(v <= 0 for v in vals):
            return None
        sd_t, sd_c, n_t, n_c, mean_t, mean_c = vals
        return (sd_t**2 / (n_t * mean_t**2)) + (sd_c**2 / (n_c * mean_c**2))
    except (TypeError, ValueError):
        return None


def get_sd(row):
    sd_t = row.get("sd_treatment")
    sd_c = row.get("sd_control")
    n_t = row.get("treatment_n") or row.get("control_n")
    n_c = row.get("control_n") or row.get("treatment_n")

    if pd.isna(sd_t) and not pd.isna(row.get("se_treatment")):
        se_t = float(row["se_treatment"])
        if not pd.isna(n_t) and float(n_t) > 0:
            sd_t = se_t * math.sqrt(float(n_t))
    if pd.isna(sd_c) and not pd.isna(row.get("se_control")):
        se_c = float(row["se_control"])
        if not pd.isna(n_c) and float(n_c) > 0:
            sd_c = se_c * math.sqrt(float(n_c))

    # LSD conversion
    if pd.isna(sd_t) and not pd.isna(row.get("variance_value")):
        vtype = str(row.get("variance_type", "")).upper()
        if vtype == "LSD":
            lsd = float(row["variance_value"])
            n = float(n_t) if not pd.isna(n_t) else 3.0
            df = 2 * (n - 1)
            if df > 0:
                t_crit = stats.t.ppf(0.975, df)
                se_diff = lsd / (t_crit * math.sqrt(2))
                sd_approx = se_diff * math.sqrt(n)
                sd_t = sd_approx
                sd_c = sd_approx

    return sd_t, sd_c, n_t, n_c


def dersimonian_laird(effect_sizes, variances):
    yi = np.array(effect_sizes, dtype=float)
    vi = np.array(variances, dtype=float)
    wi = 1.0 / vi
    sum_w = np.sum(wi)
    mu_fe = np.sum(wi * yi) / sum_w
    Q = np.sum(wi * (yi - mu_fe)**2)
    k = len(yi)
    df = k - 1
    Q_p = 1 - stats.chi2.cdf(Q, df) if df > 0 else 1.0
    C = sum_w - np.sum(wi**2) / sum_w
    tau2 = max(0, (Q - df) / C) if C > 0 else 0
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0
    wi_re = 1.0 / (vi + tau2)
    sum_w_re = np.sum(wi_re)
    mu_re = np.sum(wi_re * yi) / sum_w_re
    se_re = 1.0 / np.sqrt(sum_w_re)
    ci_lo = mu_re - 1.96 * se_re
    ci_hi = mu_re + 1.96 * se_re

    return {
        "pooled_lnRR": float(mu_re),
        "pooled_pct": float(lnRR_to_pct(mu_re)),
        "se_lnRR": float(se_re),
        "ci_lo_pct": float(lnRR_to_pct(ci_lo)),
        "ci_hi_pct": float(lnRR_to_pct(ci_hi)),
        "tau2": float(tau2),
        "I2": float(I2),
        "Q": float(Q),
        "Q_df": int(df),
        "Q_p": float(Q_p),
        "k": int(k),
    }


def bootstrap_ci(values, n_boot=5000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return None, None
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def synthesize_topic(topic_dir: Path, benchmark_est: float, benchmark_ci: tuple = None,
                     benchmark_source: str = ""):
    """Run DL random-effects synthesis on validated data for a topic."""
    csv_path = topic_dir / "4_extract" / "summary_validated.csv"
    if not csv_path.exists():
        print(f"  No validated CSV for {topic_dir.name}")
        return None

    df = pd.read_csv(csv_path)
    n_obs = len(df)
    n_papers = df["paper_id"].nunique()

    # Recalculate effect_pct
    df["effect_pct"] = (
        (df["treatment_mean"] - df["control_mean"])
        / df["control_mean"].abs() * 100
    )

    # Filter yield observations (keep all that passed PICO validation)
    yield_keywords = ["yield", "grain", "biomass", "dry weight", "fresh weight",
                      "kg/ha", "t/ha", "mg/ha", "g/plant", "g/m2", "productivity",
                      "fruit", "tuber", "seed", "pod"]
    exclude_keywords = ["soil", "nitrogen content", "protein", "quality",
                        "heavy metal", "adsorption", "removal", "weed",
                        "colonization", "spore", "root length", "p uptake"]

    def is_yield(row):
        text = "|".join(str(row.get(c, "")).lower() for c in ["outcome", "outcome_unit"])
        has = any(k in text for k in yield_keywords)
        excl = any(k in text for k in exclude_keywords)
        return has and not excl

    mask = df.apply(is_yield, axis=1)
    df_yield = df[mask].copy()

    # Require positive means
    df_yield = df_yield[
        (df_yield["treatment_mean"] > 0) & (df_yield["control_mean"] > 0)
    ].copy()

    # Outlier removal
    df_yield = df_yield[
        (df_yield["effect_pct"] > -90) & (df_yield["effect_pct"] < 300)
    ].copy()

    n_yield = len(df_yield)
    n_yield_papers = df_yield["paper_id"].nunique()

    if n_yield == 0:
        print(f"  No yield observations after filtering for {topic_dir.name}")
        return None

    # Simple stats
    effs = df_yield["effect_pct"].dropna()
    simple_mean = float(effs.mean())
    simple_median = float(effs.median())
    boot_lo, boot_hi = bootstrap_ci(effs.tolist())

    # lnRR for DL
    lnRR_vals = []
    lnRR_vars = []
    for _, row in df_yield.iterrows():
        lr = compute_lnRR(row["treatment_mean"], row["control_mean"])
        if lr is None:
            continue
        sd_t, sd_c, n_t, n_c = get_sd(row)
        vr = compute_variance_lnRR(sd_t, sd_c, n_t, n_c,
                                    row["treatment_mean"], row["control_mean"])
        if vr and vr > 0:
            lnRR_vals.append(lr)
            lnRR_vars.append(vr)

    dl = None
    if len(lnRR_vals) >= 3:
        dl = dersimonian_laird(lnRR_vals, lnRR_vars)

    # Results
    dl_pct = dl["pooled_pct"] if dl else simple_mean
    dl_ci = [dl["ci_lo_pct"], dl["ci_hi_pct"]] if dl else [boot_lo, boot_hi]

    direction_match = (dl_pct < 0) == (benchmark_est < 0) if dl_pct != 0 else None
    ci_overlap = (dl_ci[0] <= benchmark_est <= dl_ci[1]) if dl_ci[0] is not None else None

    result = {
        "topic": topic_dir.name,
        "n_obs_validated": int(n_obs),
        "n_papers_validated": int(df["paper_id"].nunique()),
        "n_yield_obs": int(n_yield),
        "n_yield_papers": int(n_yield_papers),
        "simple_mean_pct": round(simple_mean, 2),
        "simple_median_pct": round(simple_median, 2),
        "bootstrap_ci_95": [round(boot_lo, 2), round(boot_hi, 2)] if boot_lo else None,
        "DL_pooled_pct": round(dl["pooled_pct"], 2) if dl else None,
        "DL_ci_95_pct": [round(dl["ci_lo_pct"], 2), round(dl["ci_hi_pct"], 2)] if dl else None,
        "DL_I2": round(dl["I2"], 1) if dl else None,
        "DL_k": dl["k"] if dl else None,
        "benchmark_pct": benchmark_est,
        "benchmark_source": benchmark_source,
        "direction_match": direction_match,
        "benchmark_in_CI": ci_overlap,
        "pct_negative": round((effs < 0).mean() * 100, 1),
        "pct_positive": round((effs > 0).mean() * 100, 1),
    }

    return result


def main():
    topics = [
        {
            "dir": ROOT / "organic_yield_gap",
            "benchmark": -19.2,
            "benchmark_ci": (-21.5, -16.8),
            "source": "Ponisio et al. 2015",
        },
        {
            "dir": ROOT / "notill_tillage",
            "benchmark": -5.7,
            "benchmark_ci": (-6.7, -4.8),
            "source": "Pittelkow et al. 2015",
        },
        {
            "dir": ROOT / "mycorrhiza_yield",
            "benchmark": 23.0,
            "benchmark_ci": None,
            "source": "Hoeksema et al. 2010",
        },
        {
            "dir": ROOT / "legume_rotation",
            "benchmark": 20.0,
            "benchmark_ci": (18.0, 22.0),
            "source": "Zhao et al. 2022",
        },
        {
            "dir": ROOT / "biochar_crop_yield",
            "benchmark": 16.0,
            "benchmark_ci": (12.0, 20.0),
            "source": "Ye et al. 2020",
        },
    ]

    print("=" * 70)
    print("RE-SYNTHESIS WITH PICO-VALIDATED DATA")
    print("=" * 70)

    all_results = []
    for topic in topics:
        if not topic["dir"].exists():
            continue
        print(f"\n--- {topic['dir'].name} ---")
        result = synthesize_topic(
            topic["dir"],
            topic["benchmark"],
            topic.get("benchmark_ci"),
            topic["source"],
        )
        if result:
            all_results.append(result)
            dl_str = f"{result['DL_pooled_pct']:+.2f}% [{result['DL_ci_95_pct'][0]:+.1f}, {result['DL_ci_95_pct'][1]:+.1f}]" if result["DL_pooled_pct"] is not None else f"{result['simple_mean_pct']:+.2f}% (simple mean)"
            print(f"  Yield obs: {result['n_yield_obs']} across {result['n_yield_papers']} papers")
            print(f"  Our pooled: {dl_str}")
            print(f"  Benchmark:  {result['benchmark_pct']:+.1f}% ({result['benchmark_source']})")
            print(f"  Direction:  {'MATCH' if result['direction_match'] else 'OPPOSITE'}")
            if result["benchmark_in_CI"] is not None:
                print(f"  Benchmark in CI: {'YES' if result['benchmark_in_CI'] else 'NO'}")

    # Summary table
    print(f"\n{'='*70}")
    print("COMPARISON: BEFORE vs AFTER PICO VALIDATION")
    print(f"{'='*70}")
    print(f"{'Topic':20s} {'Before':>12s} {'After':>12s} {'Benchmark':>12s} {'Dir':>5s} {'CI?':>4s}")
    print("-" * 70)

    before_vals = {
        "organic_yield_gap": "-3.9%",
        "notill_tillage": "+9.6%",
        "mycorrhiza_yield": "+29.2%",
        "legume_rotation": "?",
        "biochar_crop_yield": "?",
    }
    for r in all_results:
        after = f"{r['DL_pooled_pct']:+.1f}%" if r["DL_pooled_pct"] is not None else f"{r['simple_mean_pct']:+.1f}%"
        before = before_vals.get(r["topic"], "?")
        bench = f"{r['benchmark_pct']:+.1f}%"
        d = "Y" if r["direction_match"] else "N"
        ci = "Y" if r.get("benchmark_in_CI") else "N"
        print(f"{r['topic']:20s} {before:>12s} {after:>12s} {bench:>12s} {d:>5s} {ci:>4s}")

    # Write results
    out_path = ROOT / "resynthesis_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
