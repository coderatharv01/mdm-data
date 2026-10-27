"""
analysis.py
-----------
All statistical calculations for ReviewAnalytics.
Each function receives a pandas DataFrame and user-chosen column names,
performs the calculation, generates charts, and returns a structured dict
that Flask sends to the frontend as JSON.
"""

import numpy as np
import pandas as pd
from scipy import stats

import charts  # Our chart-generation module


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_numeric(series):
    """Drop NaN and convert to float. Returns a clean numpy array."""
    return pd.to_numeric(series, errors="coerce").dropna().astype(float)


def _interpret_r(r):
    """Return a human-readable strength label for a Pearson/Spearman r value."""
    abs_r = abs(r)
    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "very weak (negligible)"
    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"


def _safe_float(val):
    """Convert numpy scalars or Python floats to plain Python float safely."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None


# ── 1. Mean ──────────────────────────────────────────────────────────────────

def calculate_mean(df, column):
    """
    Calculate the arithmetic mean of a single numerical column.
    Returns mean, std dev, min, max, count, and a histogram chart.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}

    series = _clean_numeric(df[column])
    if len(series) < 2:
        return {"error": f"Column '{column}' has fewer than 2 valid numeric values."}

    mean_val = float(series.mean())
    median_val = float(series.median())
    std_val = float(series.std())
    min_val = float(series.min())
    max_val = float(series.max())
    count = int(len(series))

    chart_b64 = charts.make_histogram(
        series.tolist(),
        title=f"Distribution of {column}",
        xlabel=column,
        ylabel="Frequency",
        mean_val=mean_val,
        median_val=median_val,
    )

    return {
        "analysis_type": "mean",
        "column": column,
        "results": {
            "mean": _safe_float(mean_val),
            "std_dev": _safe_float(std_val),
            "min": _safe_float(min_val),
            "max": _safe_float(max_val),
            "valid_observations": count,
        },
        "interpretation": (
            f"The arithmetic mean of '{column}' is {mean_val:.4f}, "
            f"calculated from {count} valid observations. "
            f"The standard deviation is {std_val:.4f}, indicating the spread around the mean. "
            f"Values range from {min_val} to {max_val}."
        ),
        "chart": chart_b64,
    }


# ── 2. Median ────────────────────────────────────────────────────────────────

def calculate_median(df, column):
    """
    Calculate the median of a single numerical column.
    Also shows Q1, Q3, and IQR for context.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}

    series = _clean_numeric(df[column])
    if len(series) < 2:
        return {"error": f"Column '{column}' has fewer than 2 valid numeric values."}

    median_val = float(series.median())
    mean_val = float(series.mean())
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    count = int(len(series))

    chart_b64 = charts.make_histogram(
        series.tolist(),
        title=f"Distribution of {column}",
        xlabel=column,
        ylabel="Frequency",
        mean_val=mean_val,
        median_val=median_val,
    )

    skew_desc = (
        "approximately symmetric"
        if abs(mean_val - median_val) < 0.1 * (series.max() - series.min() + 1e-9)
        else ("positively skewed (right tail)" if mean_val > median_val else "negatively skewed (left tail)")
    )

    return {
        "analysis_type": "median",
        "column": column,
        "results": {
            "median": _safe_float(median_val),
            "mean": _safe_float(mean_val),
            "q1": _safe_float(q1),
            "q3": _safe_float(q3),
            "iqr": _safe_float(iqr),
            "valid_observations": count,
        },
        "interpretation": (
            f"The median of '{column}' is {median_val:.4f} (n = {count}). "
            f"The mean is {mean_val:.4f}. "
            f"The interquartile range (IQR) is {iqr:.4f} (Q1 = {q1:.4f}, Q3 = {q3:.4f}). "
            f"The distribution appears {skew_desc}."
        ),
        "chart": chart_b64,
    }


# ── 3. Frequency Distribution ────────────────────────────────────────────────

def calculate_frequency(df, column):
    """
    Count occurrences of each unique value in a column.
    Works for both categorical and numerical columns.
    Returns frequency table, percentages, bar chart, and pie chart.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}

    # Drop rows where the selected column is empty
    series = df[column].dropna().astype(str).str.strip()
    series = series[series != ""]
    if len(series) == 0:
        return {"error": f"Column '{column}' contains no valid data."}

    counts = series.value_counts()
    total = len(series)
    percentages = (counts / total * 100).round(2)

    labels = counts.index.tolist()
    freq_values = counts.values.tolist()
    pct_values = percentages.values.tolist()

    # Frequency table rows
    table_rows = [
        {"category": lbl, "count": int(cnt), "percentage": float(pct)}
        for lbl, cnt, pct in zip(labels, freq_values, pct_values)
    ]

    bar_chart = charts.make_bar_chart(
        labels, freq_values,
        title=f"Frequency Distribution — {column}",
        xlabel=column,
        ylabel="Count",
    )

    pie_chart = charts.make_pie_chart(
        labels, freq_values,
        title=f"Proportion — {column}",
    )

    mode_val = counts.index[0]
    mode_count = int(counts.iloc[0])

    return {
        "analysis_type": "frequency",
        "column": column,
        "results": {
            "total_responses": total,
            "unique_categories": len(labels),
            "mode": str(mode_val),
            "mode_count": mode_count,
            "table": table_rows,
        },
        "interpretation": (
            f"The frequency distribution of '{column}' shows {len(labels)} unique categories "
            f"across {total} valid responses. "
            f"The most common value is '{mode_val}' with {mode_count} occurrences "
            f"({(mode_count / total * 100):.1f}% of responses)."
        ),
        "chart": bar_chart,
        "pie_chart": pie_chart,
    }


# ── 4. Correlation ───────────────────────────────────────────────────────────

def calculate_correlation(df, col_x, col_y, method="pearson"):
    """
    Calculate Pearson or Spearman correlation between two numerical columns.
    Returns coefficient, p-value, scatter plot, and interpretation.
    """
    for col in [col_x, col_y]:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found in dataset."}

    # Use only rows where BOTH columns have valid numeric data
    temp = df[[col_x, col_y]].copy()
    temp[col_x] = pd.to_numeric(temp[col_x], errors="coerce")
    temp[col_y] = pd.to_numeric(temp[col_y], errors="coerce")
    temp = temp.dropna()

    if len(temp) < 3:
        return {"error": "Fewer than 3 valid paired observations — correlation cannot be computed."}

    x = temp[col_x].values.astype(float)
    y = temp[col_y].values.astype(float)

    # Check for constant columns (zero variance)
    if np.std(x) == 0:
        return {"error": f"Column '{col_x}' is constant — correlation is undefined."}
    if np.std(y) == 0:
        return {"error": f"Column '{col_y}' is constant — correlation is undefined."}

    method = method.lower()
    if method == "spearman":
        r_stat, p_val = stats.spearmanr(x, y)
        method_label = "Spearman rank"
    else:
        r_stat, p_val = stats.pearsonr(x, y)
        method_label = "Pearson"

    r_val = float(r_stat)
    p_value = float(p_val)
    n = len(temp)

    scatter_chart = charts.make_scatter(
        x.tolist(), y.tolist(),
        title=f"{method_label} Correlation — {col_x} vs {col_y}",
        xlabel=col_x,
        ylabel=col_y,
    )

    significance = (
        f"The correlation is statistically significant (p = {p_value:.4f} < 0.05)."
        if p_value < 0.05
        else f"The correlation is NOT statistically significant (p = {p_value:.4f} ≥ 0.05)."
    )

    strength_label = _interpret_r(r_val)

    return {
        "analysis_type": "correlation",
        "columns": [col_x, col_y],
        "method": method_label,
        "results": {
            "correlation_coefficient": _safe_float(r_val),
            "p_value": _safe_float(p_value),
            "valid_observations": n,
        },
        "interpretation": (
            f"The {method_label} correlation coefficient between '{col_x}' and '{col_y}' "
            f"is r = {r_val:.4f} (n = {n}). "
            f"This indicates a {strength_label} relationship. "
            f"{significance} "
            f"Note: correlation does not imply causation."
        ),
        "chart": scatter_chart,
    }


# ── 5. Regression Analysis ───────────────────────────────────────────────────

def calculate_regression(df, dep_col, indep_col):
    """
    Simple ordinary least squares linear regression using SciPy.
    dep_col   : dependent variable (Y)
    indep_col : independent variable (X)
    """
    for col in [dep_col, indep_col]:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found in dataset."}

    temp = df[[dep_col, indep_col]].copy()
    temp[dep_col] = pd.to_numeric(temp[dep_col], errors="coerce")
    temp[indep_col] = pd.to_numeric(temp[indep_col], errors="coerce")
    temp = temp.dropna()

    if len(temp) < 3:
        return {"error": "Fewer than 3 valid paired observations — regression cannot be computed."}

    x = temp[indep_col].values.astype(float)
    y = temp[dep_col].values.astype(float)

    if np.std(x) == 0:
        return {"error": f"Column '{indep_col}' is constant — regression is undefined."}
    if np.std(y) == 0:
        return {
            "error": (
                f"Column '{dep_col}' is constant — there is nothing for regression to explain. "
                "This may also indicate that the column is categorical rather than numerical."
            )
        }

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    n = len(temp)

    # Generate scatter plot with regression line
    chart_b64 = charts.make_scatter(
        x.tolist(), y.tolist(),
        title=f"Regression: {dep_col} ~ {indep_col}",
        xlabel=indep_col,
        ylabel=dep_col,
        regression_line={"slope": slope, "intercept": intercept},
    )

    sign = "+" if intercept >= 0 else "-"
    equation = f"{dep_col} = {slope:.4f} × {indep_col} {sign} {abs(intercept):.4f}"

    significance = (
        f"The slope is statistically significant (p = {float(p_value):.4f} < 0.05)."
        if float(p_value) < 0.05
        else f"The slope is NOT statistically significant (p = {float(p_value):.4f} ≥ 0.05)."
    )

    r2_pct = float(r_squared) * 100

    return {
        "analysis_type": "regression",
        "columns": {"dependent": dep_col, "independent": indep_col},
        "results": {
            "slope": _safe_float(slope),
            "intercept": _safe_float(intercept),
            "r_squared": _safe_float(r_squared),
            "r_value": _safe_float(r_value),
            "p_value": _safe_float(p_value),
            "std_err": _safe_float(std_err),
            "valid_observations": n,
            "equation": equation,
        },
        "interpretation": (
            f"Simple linear regression predicts '{dep_col}' from '{indep_col}' using {n} observations.\n"
            f"Regression equation: {equation}.\n"
            f"The slope ({slope:.4f}) means that for each 1-unit increase in '{indep_col}', "
            f"'{dep_col}' changes by {slope:.4f} units on average.\n"
            f"R² = {r_squared:.4f}, meaning '{indep_col}' explains {r2_pct:.1f}% of the variance in '{dep_col}'.\n"
            f"{significance}"
        ),
        "chart": chart_b64,
    }


# ── 6. Chi-square Test ───────────────────────────────────────────────────────

def calculate_chi_square(df, col1, col2, alpha=0.05):
    """
    Chi-square test of independence between two categorical variables.
    H₀: The two variables are independent.
    H₁: The two variables are associated (not independent).
    """
    for col in [col1, col2]:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found in dataset."}

    temp = df[[col1, col2]].dropna()
    temp[col1] = temp[col1].astype(str).str.strip()
    temp[col2] = temp[col2].astype(str).str.strip()
    temp = temp[(temp[col1] != "") & (temp[col2] != "")]

    if len(temp) < 5:
        return {"error": "Fewer than 5 valid rows — Chi-square test requires more data."}

    # Build contingency table
    contingency = pd.crosstab(temp[col1], temp[col2])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            "error": (
                "The contingency table must have at least 2 rows and 2 columns. "
                "One of the selected columns may have only one unique value."
            )
        }

    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Check expected cell frequencies
    low_expected = float((expected < 5).mean() * 100)
    warning = None
    if low_expected > 20:
        warning = (
            f"{low_expected:.1f}% of expected cell frequencies are below 5. "
            "The Chi-square approximation may be unreliable. "
            "Consider Fisher's Exact Test (available for 2×2 tables) or "
            "combining sparse categories."
        )

    # Convert tables to JSON-friendly format
    observed_dict = contingency.to_dict(orient="split")
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    expected_dict = expected_df.round(2).to_dict(orient="split")

    # Flatten for grouped bar chart
    row_labels = contingency.index.tolist()
    col_labels = contingency.columns.tolist()
    flat_categories = [f"{r} | {c}" for r in row_labels for c in col_labels]
    flat_observed = [int(contingency.loc[r, c]) for r in row_labels for c in col_labels]
    flat_expected = [round(float(expected_df.loc[r, c]), 2) for r in row_labels for c in col_labels]

    chart_b64 = charts.make_grouped_bar(
        flat_categories, flat_observed, flat_expected,
        title=f"Observed vs Expected — {col1} × {col2}",
    )

    significant = bool(p_value < alpha)

    if significant:
        conclusion = (
            f"At the {alpha} significance level, we REJECT the null hypothesis. "
            f"There is statistically significant evidence of an association between "
            f"'{col1}' and '{col2}' (χ² = {chi2_stat:.4f}, df = {dof}, p = {p_value:.4f}). "
            f"Note: This shows an association, not necessarily causation."
        )
    else:
        conclusion = (
            f"At the {alpha} significance level, we FAIL TO REJECT the null hypothesis. "
            f"There is insufficient evidence of an association between '{col1}' and '{col2}' "
            f"(χ² = {chi2_stat:.4f}, df = {dof}, p = {p_value:.4f})."
        )

    return {
        "analysis_type": "chi_square",
        "columns": [col1, col2],
        "results": {
            "chi2_statistic": _safe_float(chi2_stat),
            "p_value": _safe_float(p_value),
            "degrees_of_freedom": int(dof),
            "alpha": alpha,
            "significant": significant,
            "low_expected_pct": round(low_expected, 1),
            "observed_table": observed_dict,
            "expected_table": expected_dict,
        },
        "warning": warning,
        "interpretation": (
            f"Chi-square test of independence between '{col1}' and '{col2}'.\n"
            f"H₀: The two variables are independent.\n"
            f"Observed n = {len(temp)}.\n"
            f"{conclusion}"
        ),
        "chart": chart_b64,
    }


# ── Summary statistics (for the dashboard cards) ─────────────────────────────

def calculate_summary(df):
    """
    Calculate the four summary card metrics.
    Gracefully handles missing columns.
    """
    total = int(len(df))
    summary = {"total_responses": total}

    if "review_rating" in df.columns:
        ratings = _clean_numeric(df["review_rating"])
        if len(ratings) > 0:
            summary["avg_rating"] = round(float(ratings.mean()), 2)
            summary["median_rating"] = round(float(ratings.median()), 2)

    if "purchase_intention" in df.columns:
        pi = df["purchase_intention"].dropna().astype(str).str.strip().str.lower()
        likely = (pi == "likely").sum()
        if len(pi) > 0:
            summary["pct_likely"] = round(float(likely / len(pi) * 100), 1)

    return summary
