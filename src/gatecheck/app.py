"""Streamlit dashboard. Every number it shows comes from results/metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from .config import BusinessAssumptions, Config, DecisionRules
from .decision import MetricEvidence, evaluate_experiment, financial_impact
from .frequentist import newcombe_interval

VERDICT_COLOURS = {"ship": "#3f7a52", "hold": "#8a6d3b", "roll back": "#a8324a"}


def load_metrics(path: Path) -> dict:
    """Read a completed run. The dashboard never re-runs the pipeline itself."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `gatecheck run` first")
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_from_metrics(metrics: dict, alpha: float = 0.05) -> MetricEvidence:
    """Pull the six numbers the verdict depends on out of the metrics file.

    The interval is recomputed from the stored counts rather than read off, because
    the dashboard lets the reader move the significance level and an interval fixed
    at 95% would then be tested against a threshold it does not match.
    """
    primary = metrics["dataset"]["primary_metric"]
    freq = metrics["frequentist"][primary]
    low, high = newcombe_interval(
        freq["successes_control"],
        freq["n_control"],
        freq["successes_treatment"],
        freq["n_treatment"],
        alpha,
    )
    return MetricEvidence(
        metric=primary,
        absolute_diff_pp=freq["absolute_diff_pp"],
        ci_low_pp=low * 100.0,
        ci_high_pp=high * 100.0,
        p_value=freq["p_value"],
        posterior_prob_treatment_better=metrics["bayes"]["prob_treatment_better"],
        prob_worse_than_floor=metrics["bayes"]["prob_worse_than_floor"],
    )


def main() -> None:  # pragma: no cover - exercised by hand, logic lives in decision.py
    config = Config()
    st.set_page_config(page_title="gatecheck", layout="wide")
    st.title("Gate placement experiment review")

    try:
        metrics = load_metrics(config.paths.results / "metrics.json")
    except FileNotFoundError as error:
        st.error(str(error))
        return

    with st.sidebar:
        st.header("Decision rules")
        alpha = st.slider("significance level", 0.01, 0.10, 0.05, 0.01)
        posterior_threshold = st.slider(
            "posterior probability needed to ship", 0.80, 0.995, 0.95, 0.005
        )
        floor = st.slider("day-7 retention the team will trade away (pp)", -1.0, 0.0, -0.20, 0.05)
        st.header("Business assumptions")
        monthly = st.number_input("new players a month", 100_000, 10_000_000, 1_500_000, 100_000)
        arpdau = st.number_input("ARPDAU (USD)", 0.01, 1.00, 0.085, 0.005, format="%.3f")
        days = st.number_input("monetised days per retained player", 1.0, 120.0, 21.0, 1.0)
        rollout = st.slider("share of players under the change", 0.0, 1.0, 1.0, 0.05)

    evidence = evidence_from_metrics(metrics, alpha)
    rules = DecisionRules(
        alpha=alpha, min_posterior_prob_to_ship=posterior_threshold, practical_floor_pp=floor
    )
    business = BusinessAssumptions(
        monthly_new_players=int(monthly),
        arpdau_usd=arpdau,
        days_monetised_per_retained_player=days,
        variant_rollout_share=rollout,
    )
    verdict = evaluate_experiment(evidence, rules, metrics["validity"]["passed"])
    money = financial_impact(
        evidence.absolute_diff_pp, evidence.ci_low_pp, evidence.ci_high_pp, business
    )

    colour = VERDICT_COLOURS.get(verdict.recommendation, "#444444")
    st.markdown(
        f"<h2 style='color:{colour};margin-bottom:0'>{verdict.recommendation.upper()}</h2>"
        f"<p style='color:#666'>{verdict.confidence} confidence</p>",
        unsafe_allow_html=True,
    )
    for reason in verdict.reasons:
        st.write(f"- {reason}")

    left, middle, right = st.columns(3)
    left.metric("day-7 retention change", f"{evidence.absolute_diff_pp:+.2f} pp")
    middle.metric(
        "chance the change helps", f"{evidence.posterior_prob_treatment_better * 100:.2f}%"
    )
    right.metric("annual revenue effect", f"${money.annual_revenue_delta_usd:,.0f}")

    st.subheader("Revenue range implied by the confidence interval")
    st.write(
        f"${money.annual_revenue_delta_low_usd:,.0f} to "
        f"${money.annual_revenue_delta_high_usd:,.0f} "
        f"a year, on {money.annual_new_players:,.0f} players reaching the gate."
    )
    with st.expander("assumptions behind that number"):
        for assumption in money.assumptions:
            st.write(f"- {assumption}")

    st.subheader("Evidence")
    st.json(
        {
            "frequentist": metrics["frequentist"],
            "bayes": metrics["bayes"],
            "peeking": metrics["peeking"]["study"],
            "targeting": metrics["targeting"],
        },
        expanded=False,
    )
    figure_names = (
        "retention_by_arm.png",
        "posterior_day7.png",
        "peeking_cost.png",
        "bandit_regret.png",
    )
    for name in figure_names:
        figure = config.paths.assets / name
        if figure.exists():
            st.image(str(figure))


if __name__ == "__main__":  # pragma: no cover
    main()
