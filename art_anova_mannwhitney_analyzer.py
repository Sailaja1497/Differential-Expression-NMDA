

"""

Author: Sailaja Kuruvada
Date: 2025


ART ANOVA + Post-hoc Mann–Whitney U on aligned-ranked data - LOOP VERSION

This script performs Aligned Rank Transform (ART) ANOVA followed by post-hoc Mann-Whitney U tests
on aligned-ranked data. It runs across all CSV files in a specified folder and includes filename
in results for comprehensive analysis.

Key Features:
- ART ANOVA (Type II) on aligned-ranked data for non-parametric factorial analysis
- Post-hoc Mann-Whitney U tests with multiple testing correction
- Support for up to 3-way factorial designs
- Batch processing of multiple CSV files
- Comprehensive output including individual and combined results
- Effect size calculations (Cliff's delta and r effect size)

Required Libraries and Versions:
- pandas>=1.3.0: Data manipulation and analysis library for organizing data (pd.read_csv), creating DataFrames, and exporting results to CSV format (df.to_csv)
- numpy>=1.20.0: Fundamental package for scientific computing, providing numerical operations (np.nan), array handling, and statistical computations throughout the analysis pipeline
- scipy>=1.7.0: Scientific computing library providing statistical functions including ranking (rankdata) and non-parametric tests (mannwhitneyu) for the ART procedure
- statsmodels>=0.13.0: Statistical modeling library for linear models (ols), ANOVA analysis (anova_lm), and multiple testing correction (multipletests) including Holm and FDR methods
- itertools: Standard library for generating factor combinations (combinations) to analyze all possible factorial effects
- typing: Standard library for type hints (List, Dict, Tuple, Optional) to improve code readability and IDE support
- os: Standard library for file path operations (os.path.join, os.path.basename), directory creation (os.makedirs), and file system operations
- glob: Standard library for pattern-based file searching (glob.glob) to find all CSV files in the data directory
- math: Standard library for mathematical operations including square root (math.sqrt) in effect size calculations

Factors (categorical): e.g., Genotype, Region, Subunit
Response (numeric): e.g., Integrated Density, Mean Intensity, etc.

OUTPUT RESULTS:
- CSV files: Individual 'ART_ANOVA_*.csv' files for each dataset
  Contains: ART ANOVA results for each input CSV file
  Columns: Source, SS, DF, MS, F, p_value, p_uncorrected, significant
  Format: One row per statistical effect (main effects, interactions)
- CSV file: 'Combined_ART_ANOVA_results.csv' (all files combined)
  Contains: Combined ART ANOVA results from all processed datasets
  Columns: Filename, Source, SS, DF, MS, F, p_value, p_uncorrected, significant
  Format: One row per statistical effect per dataset
- Directory: 'art_outputs_mann_loop/' containing individual results per file
  Contains: Detailed statistical analysis outputs for each dataset
  Includes: ANOVA tables, post-hoc test results, effect sizes, multiple testing corrections
"""

import os                             # Used for file path operations (os.path.join, os.path.basename), directory creation (os.makedirs), and file system operations
import math                           # Used for mathematical operations including square root (math.sqrt) in effect size calculations
import pandas as pd                   # Used for data manipulation (pd.read_csv), DataFrame operations, and CSV export (df.to_csv) for statistical results
import numpy as np                    # Used for numerical operations (np.nan), array handling, and statistical computations throughout the analysis pipeline
from itertools import combinations    # Used to generate all possible factor combinations for factorial ANOVA effects (main effects, 2-way, 3-way interactions)
from typing import List, Dict, Tuple, Optional  # Used for type hints to improve code readability and IDE support for complex data structures
from scipy.stats import rankdata, mannwhitneyu  # Used for ranking data (rankdata) in ART procedure and non-parametric pairwise comparisons (mannwhitneyu)
from statsmodels.formula.api import ols         # Used for fitting linear models (ols) in the ART alignment and ANOVA procedures
from statsmodels.stats.anova import anova_lm    # Used for performing ANOVA analysis (anova_lm) on ranked data with Type II sums of squares
from statsmodels.stats.multitest import multipletests  # Used for multiple testing correction (multipletests) including Holm and FDR methods
import glob                           # Used for pattern-based file searching (glob.glob) to find all CSV files in the data directory

# ================================
# Core ART utilities
# ================================

def _effect_terms(factors: List[str]) -> List[str]:
    """Generate all effect terms (main ... k-way) in statsmodels C(...) syntax."""
    effects = []
    for i in range(1, len(factors) + 1):
        for combo in combinations(factors, i):
            effects.append(":".join([f"C({f})" for f in combo]))
    return effects

def _build_formula(response: str, terms: List[str]) -> str:
    if not terms:
        return f"{response} ~ 1"
    return f"{response} ~ " + " + ".join(terms)

def art_align_rank_for_effect(
    df: pd.DataFrame,
    response: str,
    factors: List[str],
    effect: str,
    ties_method: str = "average"
) -> pd.DataFrame:
    """
    For a target effect, perform canonical ART alignment and add a `_ranked_` column.
    Alignment: y - yhat_reduced + grand_mean(y), where reduced excludes `effect`.
    """
    df = df.copy()
    for f in factors:
        df[f] = df[f].astype("category")
    df[response] = df[response].astype(float)

    all_effects = _effect_terms(factors)
    reduced_terms = [t for t in all_effects if t != effect]
    formula_reduced = _build_formula(response, reduced_terms)

    try:
        mod_reduced = ols(formula_reduced, data=df).fit()
        aligned = df[response] - mod_reduced.fittedvalues + df[response].mean()
    except Exception:
        aligned = df[response].copy()

    df["_ranked_"] = rankdata(aligned, method=ties_method)
    return df

def art_anova(
    df: pd.DataFrame,
    response: str,
    factors: List[str]
) -> pd.DataFrame:
    """
    ART ANOVA (Type II on ranks). Returns tidy table with Effect, df_effect, df_resid, F, p.
    """
    df = df.copy()
    for f in factors:
        df[f] = df[f].astype("category")
    df[response] = df[response].astype(float)

    all_effects = _effect_terms(factors)
    rows = []
    for effect in all_effects:
        df_ranked = art_align_rank_for_effect(df, response, factors, effect)
        try:
            mod_rank = ols(f"_ranked_ ~ {effect}", data=df_ranked).fit()
            aov = anova_lm(mod_rank, typ=2)
            if effect in aov.index:
                row = aov.loc[effect]
                rows.append({
                    "Effect": effect.replace("C(","").replace(")",""),
                    "df_effect": int(row["df"]),
                    "df_resid": int(aov.loc["Residual","df"]),
                    "F": float(row["F"]),
                    "p": float(row["PR(>F)"])
                })
            else:
                rows.append({"Effect": effect.replace("C(","").replace(")",""),
                             "df_effect": np.nan, "df_resid": np.nan, "F": np.nan, "p": np.nan})
        except Exception:
            rows.append({"Effect": effect.replace("C(","").replace(")",""),
                         "df_effect": np.nan, "df_resid": np.nan, "F": np.nan, "p": np.nan})

    # Preserve natural order (mains, 2-ways, 3-way, ...)
    order_index = {}
    flat = [e.replace("C(","").replace(")","") for e in all_effects]
    for i, name in enumerate(flat):
        order_index[name] = i
    out = pd.DataFrame(rows)
    out["__ord__"] = out["Effect"].map(order_index)
    out = out.sort_values("__ord__").drop(columns="__ord__").reset_index(drop=True)
    return out

# ================================
# Mann–Whitney helpers
# ================================

def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta (robust effect size)."""
    # Efficient approximation using ranks: delta = (2*AUC - 1)
    # Where AUC = U/(n1*n2) for Mann–Whitney
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    auc = u_stat / (n1 * n2)
    return 2 * auc - 1

def _r_effect_size_from_u(u: float, n1: int, n2: int) -> float:
    """Compute |r| effect size from U via normal approximation."""
    mean_u = n1 * n2 / 2.0
    sd_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sd_u == 0:
        return np.nan
    z = (u - mean_u) / sd_u
    return abs(z) / math.sqrt(n1 + n2)

def mannwhitney_table(
    values: np.ndarray, groups: np.ndarray, alpha: float = 0.05, msc_method: str = "holm"
) -> Optional[pd.DataFrame]:
    """
    Pairwise Mann–Whitney U across all group levels.
    Multiple-testing correction: 'holm' or 'fdr_bh'
    Returns a tidy DataFrame or None if <2 groups.
    """
    grouplabels = pd.Series(groups).astype(str)
    levels = sorted(grouplabels.unique())
    if len(levels) < 2:
        return None

    rows = []
    for a, b in combinations(levels, 2):
        xa = values[grouplabels == a].astype(float)
        xb = values[grouplabels == b].astype(float)
        if len(xa) == 0 or len(xb) == 0:
            continue
        u, p = mannwhitneyu(xa, xb, alternative="two-sided", method="auto")
        r = _r_effect_size_from_u(u, len(xa), len(xb))
        cd = _cliffs_delta(xa, xb)
        rows.append({"group_A": a, "group_B": b, "U": u, "p": p, "|r|": r, "cliffs_delta": cd,
                     "n_A": len(xa), "n_B": len(xb)})

    if not rows:
        return None

    tab = pd.DataFrame(rows)
    # Multiple testing correction
    reject, p_adj, _, _ = multipletests(tab["p"].values, method=msc_method, alpha=alpha)
    tab["p_adj"] = p_adj
    tab["reject_H0"] = reject
    tab = tab[["group_A","group_B","n_A","n_B","U","p","p_adj","reject_H0","|r|","cliffs_delta"]]
    return tab

# ================================
# Post-hoc (Mann–Whitney) on ART-ranked data
# ================================

def _combine_label(df: pd.DataFrame, cols: List[str], sep=":") -> pd.Series:
    """Create combined label like 'A=a1:B=b2' for interaction post-hoc."""
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out.str.cat(df[c].astype(str), sep=sep)
    return out

def posthoc_art_mannwhitney(
    df: pd.DataFrame,
    response: str,
    factors: List[str],
    anova_table: pd.DataFrame,
    alpha: float = 0.05,
    msc_method: str = "holm",
    interaction_mode: str = "simple"  # "simple" or "combined"
) -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
    """
    Run Mann–Whitney U post-hoc on ART-ranked data for significant effects.
    Returns: effect_name -> list of (contrast_label, table_df)
    """
    sig_effects = anova_table.loc[anova_table["p"] < alpha, "Effect"].tolist()
    results: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}

    def parse_effect(eff: str) -> List[str]:
        return eff.split(":")

    for eff in sig_effects:
        parts = parse_effect(eff)
        effect_c = ":".join([f"C({p})" for p in parts])

        # Align + rank for this effect
        df_al = art_align_rank_for_effect(df, response, factors, effect_c)
        items: List[Tuple[str, pd.DataFrame]] = []

        if len(parts) == 1:
            factor = parts[0]
            tab = mannwhitney_table(df_al["_ranked_"].values, df_al[factor].values,
                                    alpha=alpha, msc_method=msc_method)
            if tab is not None:
                items.append(("overall", tab))

        elif len(parts) == 2:
            A, B = parts
            if interaction_mode == "combined":
                groups = _combine_label(df_al, [A, B]).values
                tab = mannwhitney_table(df_al["_ranked_"].values, groups,
                                        alpha=alpha, msc_method=msc_method)
                if tab is not None:
                    items.append(("combined A:B", tab))
            else:
                # simple effects of A | B=b  and B | A=a
                for fixed, vary in [(B, A), (A, B)]:
                    for lvl in df_al[fixed].cat.categories:
                        sl = df_al[df_al[fixed] == lvl]
                        if sl[vary].nunique() < 2: # need at least 2 groups
                            continue
                        tab = mannwhitney_table(sl["_ranked_"].values, sl[vary].values,
                                                alpha=alpha, msc_method=msc_method)
                        if tab is not None:
                            items.append((f"simple: {vary} | {fixed}={lvl}", tab))

        else:  # 3-way
            A, B, C = parts
            if interaction_mode == "combined":
                groups = _combine_label(df_al, [A,B,C]).values
                tab = mannwhitney_table(df_al["_ranked_"].values, groups,
                                        alpha=alpha, msc_method=msc_method)
                if tab is not None:
                    items.append(("combined A:B:C", tab))
            else:
                # simple-simple: A:B within each C
                for lvl in df_al[C].cat.categories:
                    sl = df_al[df_al[C] == lvl].copy()
                    if sl.shape[0] < 2:
                        continue
                    sl["AB"] = _combine_label(sl, [A,B])
                    if sl["AB"].nunique() < 2:
                        continue
                    tab = mannwhitney_table(sl["_ranked_"].values, sl["AB"].values,
                                            alpha=alpha, msc_method=msc_method)
                    if tab is not None:
                        items.append((f"simple-simple: A:B | {C}={lvl}", tab))

        if items:
            results[eff] = items

    return results

# ================================
# Main loop function
# ================================

def run_anova_on_file(filepath: str, factors: List[str], response: str, 
                      alpha: float = 0.05, mtc_method: str = "holm", 
                      interaction_mode: str = "simple") -> Dict:
    """
    Run ART ANOVA on a single file and return results with filename.
    """
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")
    
    try:
        # Load data
        df = pd.read_csv(filepath)
        for f in factors:
            df[f] = df[f].astype("category")
        df[response] = df[response].astype(float)
        
        print(f"Data shape: {df.shape}")
        print(f"Factors: {factors}")
        print(f"Response: {response}")
        
        # Run ANOVA
        anova_tbl = art_anova(df, response=response, factors=factors)
        
        # Add filename column
        anova_tbl['Filename'] = filename
        
        # Check for significant effects
        sig_tbl = anova_tbl[anova_tbl["p"] < alpha].copy()
        
        print(f"\n=== ART ANOVA Results ===")
        print(anova_tbl.round(4))
        
        if sig_tbl.empty:
            print(f"\nNo significant effects found (p < {alpha})")
            posthoc_results = {}
        else:
            print(f"\n=== Significant Effects (p < {alpha}) ===")
            print(sig_tbl.round(4))
            
            # Run post-hoc tests
            posthoc_results = posthoc_art_mannwhitney(
                df=df, response=response, factors=factors,
                anova_table=anova_tbl, alpha=alpha,
                msc_method=mtc_method, interaction_mode=interaction_mode
            )
            
            if posthoc_results:
                print(f"\n=== Post-hoc Results ===")
                for eff, tables in posthoc_results.items():
                    for label, tbl in tables:
                        print(f"- {eff} :: {label} (n={len(tbl)})")
        
        return {
            'filename': filename,
            'anova_table': anova_tbl,
            'significant_effects': sig_tbl,
            'posthoc_results': posthoc_results,
            'data_shape': df.shape,
            'success': True
        }
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return {
            'filename': filename,
            'error': str(e),
            'success': False
        }

# ================================
# Run script
# ================================

if __name__ == "__main__":
    # Configuration
    DATA_DIR = "../data/threeway_data"
    OUT_DIR = "art_outputs_mann_loop"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    FACTORS = ["Genotype", "Region", "Subunit"]
    RESPONSE = "Response"
    ALPHA = 0.05
    MTC_METHOD = "holm"      # 'holm' (FWER) or 'fdr_bh' (FDR)
    INTERACTION_MODE = "simple"  # 'simple' or 'combined'
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    csv_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(csv_files)} CSV files in {DATA_DIR}")
    print(f"Files to process: {[os.path.basename(f) for f in csv_files]}")
    
    # Process each file
    all_results = []
    all_anova_tables = []
    
    for filepath in csv_files:
        result = run_anova_on_file(
            filepath=filepath,
            factors=FACTORS,
            response=RESPONSE,
            alpha=ALPHA,
            mtc_method=MTC_METHOD,
            interaction_mode=INTERACTION_MODE
        )
        
        all_results.append(result)
        
        if result['success']:
            all_anova_tables.append(result['anova_table'])
            
            # Save individual file results
            filename = result['filename']
            base_name = filename.replace('.csv', '')
            
            # Save ANOVA results for this file
            anova_path = os.path.join(OUT_DIR, f"ART_ANOVA_{base_name}.csv")
            result['anova_table'].to_csv(anova_path, index=False)
            print(f"Saved ANOVA results for {filename} → {anova_path}")
            
            # Save post-hoc results if any
            if result['posthoc_results']:
                for eff, tables in result['posthoc_results'].items():
                    for label, tbl in tables:
                        # Clean effect name for filename
                        clean_effect = eff.replace(':', '-')
                        clean_label = label.replace(' ', '_').replace(':', '-').replace('|', '_')
                        posthoc_path = os.path.join(OUT_DIR, f"POSTHOC_{base_name}_{clean_effect}_{clean_label}.csv")
                        tbl.to_csv(posthoc_path, index=False)
                        print(f"  Saved post-hoc: {eff} :: {label} → {posthoc_path}")
    
    # Also save combined results for convenience
    if all_anova_tables:
        combined_anova = pd.concat(all_anova_tables, ignore_index=True)
        combined_path = os.path.join(OUT_DIR, "Combined_ART_ANOVA_results.csv")
        combined_anova.to_csv(combined_path, index=False)
        print(f"\n{'='*60}")
        print(f"Combined results saved to: {combined_path}")
        print(f"{'='*60}")
        
        # Summary statistics
        print(f"\nSummary across all files:")
        print(f"Total files processed: {len(all_results)}")
        print(f"Successful analyses: {sum(1 for r in all_results if r['success'])}")
        print(f"Files with significant effects: {sum(1 for r in all_results if r['success'] and not r['significant_effects'].empty)}")
        
        # Show significant effects by file
        print(f"\nSignificant effects by file:")
        for result in all_results:
            if result['success'] and not result['significant_effects'].empty:
                sig_effects = result['significant_effects']['Effect'].tolist()
                print(f"- {result['filename']}: {', '.join(sig_effects)}")
            elif result['success']:
                print(f"- {result['filename']}: None")
            else:
                print(f"- {result['filename']}: Error - {result['error']}")
    
    else:
        print("\nNo successful analyses to combine.")
    
    print(f"\nIndividual results saved in: {OUT_DIR}/")
    print(f"Each file has its own ART_ANOVA_*.csv output file")
