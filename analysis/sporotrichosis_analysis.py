#!/usr/bin/env python3
"""SCI-ready analysis pipeline for cutaneous sporotrichosis retrospective cohort studies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


DIAGNOSIS_MAP = {
    1: "Fixed cutaneous form",
    2: "Lymphocutaneous form",
    3: "Skin disseminated form",
    "1": "Fixed cutaneous form",
    "2": "Lymphocutaneous form",
    "3": "Skin disseminated form",
    "fixed": "Fixed cutaneous form",
    "fixed cutaneous form": "Fixed cutaneous form",
    "lymphocutaneous": "Lymphocutaneous form",
    "lymphocutaneous form": "Lymphocutaneous form",
    "skin disseminated": "Skin disseminated form",
    "skin disseminated form": "Skin disseminated form",
}
DIAGNOSIS_ORDER = [
    "Fixed cutaneous form",
    "Lymphocutaneous form",
    "Skin disseminated form",
]
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sporotrichosis cohort analysis")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Output directory")
    parser.add_argument("--diagnosis-col", default="diagnostic_type")
    parser.add_argument("--site-col", default="onset_site")
    parser.add_argument("--season-col", default="season")
    parser.add_argument(
        "--covariates",
        nargs="*",
        default=[],
        help="Additional covariates for multinomial logistic and ML comparison",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def standardize_diagnosis(series: pd.Series) -> pd.Categorical:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(DIAGNOSIS_MAP)
    )
    normalized = normalized.fillna(series.where(series.isin(DIAGNOSIS_ORDER)))
    if normalized.isna().any():
        bad_values = sorted(set(series[normalized.isna()].astype(str)))
        raise ValueError(f"Unrecognized diagnostic type values: {bad_values}")
    return pd.Categorical(normalized, categories=DIAGNOSIS_ORDER, ordered=False)


def standardize_season(series: pd.Series) -> pd.Categorical:
    mapping = {
        "spring": "Spring",
        "summer": "Summer",
        "autumn": "Autumn",
        "fall": "Autumn",
        "winter": "Winter",
    }
    season = series.astype(str).str.strip().str.lower().map(mapping)
    season = season.fillna(series.where(series.isin(SEASON_ORDER)))
    if season.isna().any():
        bad_values = sorted(set(series[season.isna()].astype(str)))
        raise ValueError(f"Unrecognized season values: {bad_values}")
    return pd.Categorical(season, categories=SEASON_ORDER, ordered=True)


def save_figure_all_formats(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, transparent=True, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=300, transparent=True, bbox_inches="tight")


def plot_stacked_bar(df: pd.DataFrame, output_dir: Path) -> None:
    count_df = (
        df.groupby(["season", "diagnostic_type"], observed=False)
        .size()
        .reset_index(name="n")
    )
    total_df = count_df.groupby("season", observed=False)["n"].transform("sum")
    count_df["proportion"] = count_df["n"] / total_df

    pivot_df = count_df.pivot(index="season", columns="diagnostic_type", values="proportion").fillna(0)

    plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black"})
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(pivot_df))
    for diagnosis in DIAGNOSIS_ORDER:
        vals = pivot_df[diagnosis].values
        ax.bar(
            pivot_df.index.astype(str),
            vals,
            bottom=bottom,
            label=diagnosis,
            edgecolor="black",
        )
        bottom += vals
    ax.set_title("Distribution of Sporotrichosis Diagnostic Types by Season", color="black")
    ax.set_ylabel("Proportion of Patients", color="black")
    ax.set_xlabel("Season", color="black")
    ax.legend(title="Diagnostic Type", frameon=False)
    ax.set_ylim(0, 1)
    save_figure_all_formats(fig, output_dir / "season_diagnostic_stacked_bar")
    plt.close(fig)


def plot_trend(df: pd.DataFrame, output_dir: Path) -> None:
    trend_df = (
        df.groupby(["season", "diagnostic_type"], observed=False)
        .size()
        .reset_index(name="n")
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(
        data=trend_df,
        x="season",
        y="n",
        hue="diagnostic_type",
        style="diagnostic_type",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title("Seasonal Trend of Sporotrichosis Diagnostic Types", color="black")
    ax.set_xlabel("Season", color="black")
    ax.set_ylabel("Number of Patients", color="black")
    ax.legend(title="Diagnostic Type", frameon=False)
    for text in ax.get_legend().get_texts():
        text.set_color("black")
    save_figure_all_formats(fig, output_dir / "season_diagnostic_trend")
    plt.close(fig)


def fit_multinomial_logit(df: pd.DataFrame, features: list[str], output_dir: Path) -> pd.DataFrame:
    X = pd.get_dummies(df[features], drop_first=True)
    X = sm.add_constant(X)
    y = df["diagnostic_type"].cat.codes

    model = sm.MNLogit(y, X)
    result = model.fit(method="newton", maxiter=200, disp=False)

    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    rows = []
    class_labels = DIAGNOSIS_ORDER[1:]
    for col_idx, class_name in zip(params.columns, class_labels):
        for term in params.index:
            coef = params.loc[term, col_idx]
            ci_low = conf.loc[(f"y={col_idx}", term), 0]
            ci_high = conf.loc[(f"y={col_idx}", term), 1]
            rows.append(
                {
                    "comparison": f"{class_name} vs {DIAGNOSIS_ORDER[0]}",
                    "term": term,
                    "OR": np.exp(coef),
                    "CI95_low": np.exp(ci_low),
                    "CI95_high": np.exp(ci_high),
                    "p_value": pvals.loc[term, col_idx],
                }
            )
    or_df = pd.DataFrame(rows)
    or_df.to_csv(output_dir / "multinomial_logistic_or_table.csv", index=False)

    plot_forest(or_df[or_df["term"] != "const"].copy(), output_dir)
    return or_df


def plot_forest(or_df: pd.DataFrame, output_dir: Path) -> None:
    or_df = or_df.sort_values(["comparison", "OR"], ascending=[True, True])
    or_df["label"] = or_df["comparison"] + " | " + or_df["term"]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(or_df))))
    y_pos = np.arange(len(or_df))
    ax.errorbar(
        x=or_df["OR"],
        y=y_pos,
        xerr=[or_df["OR"] - or_df["CI95_low"], or_df["CI95_high"] - or_df["OR"]],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3,
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(or_df["label"], color="black")
    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (log scale)", color="black")
    ax.set_title("Multivariable Multinomial Logistic Regression Forest Plot", color="black")
    save_figure_all_formats(fig, output_dir / "multinomial_logistic_forest")
    plt.close(fig)


def run_ml_benchmark(df: pd.DataFrame, features: list[str], output_dir: Path, cv_folds: int, random_state: int) -> None:
    X = df[features]
    y = df["diagnostic_type"].cat.codes

    categorical_features = [col for col in features if X[col].dtype == "object" or str(X[col].dtype).startswith("category")]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough",
    )

    models: dict[str, object] = {
        "RandomForest": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", RandomForestClassifier(n_estimators=500, random_state=random_state, class_weight="balanced")),
            ]
        ),
        "LogisticRegression": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        solver="lbfgs",
                        multi_class="multinomial",
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }

    if HAS_CATBOOST:
        cat_idx = [features.index(c) for c in categorical_features]
        models["CatBoost"] = CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=0,
            auto_class_weights="Balanced",
            cat_features=cat_idx,
        )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    rows = []
    for name, model in models.items():
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "roc_auc_ovr": "roc_auc_ovr",
        }
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "accuracy_mean": np.mean(scores["test_accuracy"]),
                "accuracy_sd": np.std(scores["test_accuracy"]),
                "f1_macro_mean": np.mean(scores["test_f1_macro"]),
                "f1_macro_sd": np.std(scores["test_f1_macro"]),
                "roc_auc_ovr_mean": np.mean(scores["test_roc_auc_ovr"]),
                "roc_auc_ovr_sd": np.std(scores["test_roc_auc_ovr"]),
            }
        )

    benchmark_df = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
    benchmark_df.to_csv(output_dir / "ml_model_comparison.csv", index=False)


def prepare_data(raw_df: pd.DataFrame, diagnosis_col: str, site_col: str, season_col: str, covariates: Iterable[str]) -> pd.DataFrame:
    needed_cols = [diagnosis_col, site_col, season_col, *covariates]
    missing_cols = [c for c in needed_cols if c not in raw_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    df = raw_df.copy()
    df = df.loc[df[site_col].notna() & (df[site_col].astype(str).str.strip() != "")].copy()

    df["diagnostic_type"] = standardize_diagnosis(df[diagnosis_col])
    df["season"] = standardize_season(df[season_col])
    df["onset_site"] = df[site_col].astype(str).str.strip()

    for col in covariates:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    return df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input)
    df = prepare_data(raw_df, args.diagnosis_col, args.site_col, args.season_col, args.covariates)

    analysis_df = df[["diagnostic_type", "season", "onset_site", *args.covariates]].dropna().copy()
    analysis_df.to_csv(output_dir / "cleaned_analysis_dataset.csv", index=False)

    features = ["onset_site", *args.covariates]

    plot_stacked_bar(analysis_df, output_dir)
    plot_trend(analysis_df, output_dir)
    fit_multinomial_logit(analysis_df, features, output_dir)
    run_ml_benchmark(analysis_df, features, output_dir, args.cv_folds, args.random_state)

    print(f"Analysis completed. Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
