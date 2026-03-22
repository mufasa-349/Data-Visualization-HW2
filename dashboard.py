"""
Japan annual inbound immigration statistics — interactive dashboard.
Data: japan_immigration_statistics_inbound.csv (Kaggle-style dataset).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_FILE = Path(__file__).resolve().parent / "japan_immigration_statistics_inbound.csv"

REGION_KEYS = (
    "asia",
    "europe",
    "africa",
    "north_america",
    "south_america",
    "oceania",
)

REGION_LABELS = {
    "asia": "Asia",
    "europe": "Europe",
    "africa": "Africa",
    "north_america": "North America",
    "south_america": "South America",
    "oceania": "Oceania",
}

# UI / Bertin: one accent for magnitude series; distinct hues for signed YoY (redundant encoding)
ACCENT = "#38BDF8"
ACCENT_FILL = "rgba(56, 189, 248, 0.16)"
YOY_POS = "#38BDF8"
YOY_NEG = "#F97373"
YOY_NEU = "#94A3B8"
TEXT_PRIMARY = "#E8EAED"
TEXT_MUTED = "#B0B3B8"
GRID = "rgba(148, 163, 184, 0.22)"


def _pretty_country(name: str) -> str:
    s = name.replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s.title()


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["year"] = df["year"].astype(str).str.strip()
    df = df[df["year"].str.len() > 0].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    for c in df.columns:
        if c == "year":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df.sort_values("year").reset_index(drop=True)
    return df


def country_columns(df: pd.DataFrame) -> list[str]:
    skip = {"year", "total", *REGION_KEYS}
    return [c for c in df.columns if c not in skip]


def melt_regions(df: pd.DataFrame) -> pd.DataFrame:
    present = [k for k in REGION_KEYS if k in df.columns]
    long = df.melt(
        id_vars=["year"],
        value_vars=present,
        var_name="region",
        value_name="count",
    )
    long["region_label"] = long["region"].map(REGION_LABELS).fillna(long["region"])
    return long


def top_countries_for_year(df: pd.DataFrame, year: int, top_n: int) -> pd.DataFrame:
    row = df.loc[df["year"] == year]
    if row.empty:
        return pd.DataFrame(columns=["country", "count"])
    row = row.iloc[0]
    cols = country_columns(df)
    vals = {c: float(row[c]) for c in cols}
    s = pd.Series(vals).sort_values(ascending=False).head(top_n)
    out = s.reset_index()
    out.columns = ["country", "count"]
    out["label"] = out["country"].map(_pretty_country)
    return out


def _yoy_direction(pct: float) -> str:
    if pct > 0:
        return "Increase"
    if pct < 0:
        return "Decrease"
    return "Unchanged"


def apply_dashboard_theme(
    fig: go.Figure,
    *,
    height: int,
    title: str,
    subtitle: str | None = None,
    hovermode: str | None = "x unified",
) -> None:
    """Consistent dark-friendly typography, grid, and contrast (Gestalt similarity + WCAG-friendly ticks)."""
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += (
            f"<br><sup style='font-size:12px;color:{TEXT_MUTED}'>{subtitle}</sup>"
        )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=16, r=16, t=76 if subtitle else 60, b=16),
        title=dict(text=title_text, x=0.02, xanchor="left", font=dict(size=17)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, size=13),
        hoverlabel=dict(bgcolor="#1B1F2A", font=dict(size=13)),
        hovermode=hovermode,
        legend=dict(
            font=dict(color=TEXT_PRIMARY, size=12),
            bgcolor="rgba(14,17,23,0.85)",
            bordercolor="rgba(148,163,184,0.35)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        title_font=dict(size=13, color=TEXT_PRIMARY),
        tickfont=dict(size=11, color=TEXT_MUTED),
        gridcolor=GRID,
        zerolinecolor="rgba(148,163,184,0.35)",
    )
    fig.update_yaxes(
        title_font=dict(size=13, color=TEXT_PRIMARY),
        tickfont=dict(size=11, color=TEXT_MUTED),
        gridcolor=GRID,
        zerolinecolor="rgba(148,163,184,0.35)",
    )


def main() -> None:
    st.set_page_config(
        page_title="Japan Inbound Immigration — Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not DATA_FILE.exists():
        st.error(f"Data file not found: `{DATA_FILE}`")
        st.stop()

    df = load_data(DATA_FILE)
    y_min, y_max = int(df["year"].min()), int(df["year"].max())

    st.title("Japan — Inbound immigration")
    st.caption(
        "Source: `japan_immigration_statistics_inbound.csv` — annual totals with continent and country breakdown."
    )

    with st.sidebar:
        st.header("Filters")
        year_range = st.slider(
            "Year range",
            min_value=y_min,
            max_value=y_max,
            value=(y_min, y_max),
        )
        snapshot_year = st.select_slider(
            "Year for country ranking",
            options=list(range(y_min, y_max + 1)),
            value=y_max,
        )
        top_n = st.slider("Top N countries / categories", min_value=5, max_value=40, value=15)

    mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    dff = df.loc[mask].copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Selected period — sum of annual totals",
            f"{dff['total'].sum():,.0f}",
            help="Sum of the dataset's annual `total` over each year in the selected range (person-counts summed across years).",
        )
    with c2:
        st.metric(
            "Mean annual total",
            f"{dff['total'].mean():,.0f}",
            help="Average of `total` across years in the selected range (not the mean of people).",
        )
    with c3:
        peak = None
        if len(dff) > 0:
            peak = dff.loc[dff["total"].idxmax()]
        st.metric(
            "Peak year (total)",
            f"{int(peak['year'])}" if peak is not None else "—",
            delta=f"{peak['total']:,.0f}" if peak is not None else None,
            help="Calendar year with the largest `total` in the filtered range; delta shows that year's immigrant count (not % change).",
        )
    with c4:
        st.metric(
            "Full data range",
            f"{y_min} – {y_max}",
            help="Years available in the CSV after cleaning.",
        )

    tab1, tab2, tab3 = st.tabs(["Time series", "Continents", "Countries"])

    with tab1:
        # go.Scatter: fill under line (px.line does not support `fill` in all Plotly versions)
        fig_total = go.Figure(
            data=[
                go.Scatter(
                    x=dff["year"],
                    y=dff["total"],
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=ACCENT, width=2.5),
                    fillcolor=ACCENT_FILL,
                    name="Total",
                    hovertemplate="Year=%{x}<br>Immigrants (count)=%{y:,.0f}<extra></extra>",
                )
            ]
        )
        fig_total.update_xaxes(title_text="Year")
        fig_total.update_yaxes(title_text="Immigrants (count per year)")
        apply_dashboard_theme(
            fig_total,
            height=440,
            title="Annual total inbound immigrants",
            subtitle="Dataset field `total`: foreign nationals recorded as entering Japan that year (annual count).",
        )
        st.plotly_chart(fig_total, width="stretch")

        dff2 = dff.copy()
        dff2["yoy_pct"] = dff2["total"].pct_change() * 100
        yoy_df = dff2.dropna(subset=["yoy_pct"]).copy()
        yoy_df["direction"] = yoy_df["yoy_pct"].map(_yoy_direction)
        fig_yoy = px.bar(
            yoy_df,
            x="year",
            y="yoy_pct",
            color="direction",
            labels={
                "year": "Year",
                "yoy_pct": "YoY change (%)",
                "direction": "vs prior year",
            },
            color_discrete_map={
                "Increase": YOY_POS,
                "Decrease": YOY_NEG,
                "Unchanged": YOY_NEU,
            },
        )
        apply_dashboard_theme(
            fig_yoy,
            height=360,
            title="Year-over-year change in total immigrants",
            subtitle="Redundant encoding: bar position and hue show sign. YoY = (year − prior year) / prior year within the filtered table.",
            hovermode="x unified",
        )
        fig_yoy.add_hline(
            y=0,
            line_dash="solid",
            line_width=1,
            line_color="rgba(232,234,237,0.45)",
        )
        fig_yoy.update_traces(marker_line_width=0)
        st.plotly_chart(fig_yoy, width="stretch")
        st.caption(
            "YoY is computed on **consecutive rows in the filtered range** (not the full CSV). "
            "The first year in the range has no prior row inside the filter, so it is omitted."
        )

    with tab2:
        reg_long = melt_regions(dff)
        fig_stack = px.area(
            reg_long,
            x="year",
            y="count",
            color="region_label",
            labels={
                "year": "Year",
                "count": "Immigrants (count per year)",
                "region_label": "Region",
            },
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        apply_dashboard_theme(
            fig_stack,
            height=500,
            title="Inbound immigrants by continent",
            subtitle="Stacked areas use continent columns from the dataset (Asia, Europe, …); sum ≈ total when categories align.",
        )
        fig_stack.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        st.plotly_chart(fig_stack, width="stretch")

        share = reg_long.pivot(index="year", columns="region_label", values="count").fillna(0)
        share_pct = share.div(share.sum(axis=1), axis=0) * 100
        share_pct = share_pct.reset_index().melt(
            id_vars="year", var_name="Region", value_name="share_pct"
        )
        fig_share = px.line(
            share_pct,
            x="year",
            y="share_pct",
            color="Region",
            labels={"year": "Year", "share_pct": "Share of year total (%)"},
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        apply_dashboard_theme(
            fig_share,
            height=420,
            title="Continental shares of each year's total",
            subtitle="Each year's share sums to 100% across the six continent buckets.",
        )
        fig_share.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        st.plotly_chart(fig_share, width="stretch")

    with tab3:
        top_df = top_countries_for_year(df, snapshot_year, top_n)
        fig_bar = px.bar(
            top_df.sort_values("count"),
            x="count",
            y="label",
            orientation="h",
            labels={"count": "Immigrants (count)", "label": "Country / category"},
        )
        fig_bar.update_traces(marker_color=ACCENT, marker_line_width=0)
        apply_dashboard_theme(
            fig_bar,
            height=max(380, top_n * 28),
            title=f"{snapshot_year} — top inbound sources",
            subtitle=f"Top {top_n} country/category columns by count for the selected year (excludes continent totals).",
            hovermode="closest",
        )
        fig_bar.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_bar, width="stretch")

        st.subheader("Countries over time (heatmap, top 25 by volume)")
        heat_year = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
        cols = country_columns(df)
        totals = heat_year[cols].sum().sort_values(ascending=False).head(25).index.tolist()
        heat = heat_year[["year"] + totals].set_index("year")
        z = heat.T.values
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=z,
                x=heat.index.astype(str),
                y=[_pretty_country(c) for c in totals],
                colorscale="Viridis",
                hovertemplate="Year: %{x}<br>%{y}: %{z:,.0f}<extra></extra>",
                colorbar=dict(
                    title=dict(
                        text="Count",
                        font=dict(color=TEXT_MUTED, size=12),
                    ),
                    tickfont=dict(color=TEXT_MUTED, size=11),
                ),
            )
        )
        apply_dashboard_theme(
            fig_heat,
            height=540,
            title="Immigration heatmap (top 25 countries)",
            subtitle="Rows = countries with the largest summed counts over the filtered years; Viridis emphasizes magnitude.",
            hovermode="closest",
        )
        fig_heat.update_layout(
            xaxis_title="Year",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, width="stretch")



if __name__ == "__main__":
    main()
