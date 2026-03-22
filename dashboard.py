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
        st.metric("Selected period — sum of annual totals", f"{dff['total'].sum():,.0f}")
    with c2:
        st.metric("Mean annual total", f"{dff['total'].mean():,.0f}")
    with c3:
        peak = None
        if len(dff) > 0:
            peak = dff.loc[dff["total"].idxmax()]
        st.metric(
            "Peak year (total)",
            f"{int(peak['year'])}" if peak is not None else "—",
            delta=f"{peak['total']:,.0f}" if peak is not None else None,
        )
    with c4:
        st.metric("Full data range", f"{y_min} – {y_max}")

    tab1, tab2, tab3 = st.tabs(["Time series", "Continents", "Countries"])

    with tab1:
        fig_total = px.line(
            dff,
            x="year",
            y="total",
            markers=True,
            labels={"year": "Year", "total": "Immigrants (total)"},
        )
        fig_total.update_layout(
            hovermode="x unified",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            title="Annual total inbound immigrants",
        )
        st.plotly_chart(fig_total, use_container_width=True)

        dff2 = dff.copy()
        dff2["yoy_pct"] = dff2["total"].pct_change() * 100
        fig_yoy = px.bar(
            dff2.dropna(subset=["yoy_pct"]),
            x="year",
            y="yoy_pct",
            labels={"year": "Year", "yoy_pct": "YoY change (%)"},
        )
        fig_yoy.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            title="Year-over-year change in total immigrants (%)",
        )
        fig_yoy.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_yoy, use_container_width=True)

    with tab2:
        reg_long = melt_regions(dff)
        fig_stack = px.area(
            reg_long,
            x="year",
            y="count",
            color="region_label",
            labels={"year": "Year", "count": "Immigrants", "region_label": "Region"},
        )
        fig_stack.update_layout(
            height=480,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            title="Inbound immigrants by continent (stacked area)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

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
            labels={"year": "Year", "share_pct": "Share (%)"},
        )
        fig_share.update_layout(
            height=400,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            title="Continental shares of annual total (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_share, use_container_width=True)

    with tab3:
        top_df = top_countries_for_year(df, snapshot_year, top_n)
        fig_bar = px.bar(
            top_df.sort_values("count"),
            x="count",
            y="label",
            orientation="h",
            labels={"count": "Immigrants", "label": "Country / category"},
        )
        fig_bar.update_layout(
            height=max(360, top_n * 28),
            margin=dict(l=10, r=10, t=40, b=10),
            title=f"{snapshot_year} — top inbound ({top_n} categories)",
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

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
            )
        )
        fig_heat.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=40, b=10),
            title="Top 25 countries in selected year range (by total immigrants)",
            xaxis_title="Year",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)



if __name__ == "__main__":
    main()
