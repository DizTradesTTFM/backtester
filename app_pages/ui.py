"""Shared look for every page: dark, clean sections, plain-English labels."""

import html as _h
from contextlib import contextmanager

import streamlit as st

BLUE, AMBER, PINK, GREEN = "#4aa3ff", "#ffb020", "#ff5c8a", "#3ddc84"
TEXT, MUTED, DIM, LINE = "#f2f2f3", "#8a8a93", "#6b6b74", "#202024"
VERDICT_COLORS = {"PASS": GREEN, "WATCH": AMBER, "FAIL": PINK}
VERDICT_WORDS = {"PASS": "Ready", "WATCH": "Close", "FAIL": "Fails"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.block-container {max-width: 1180px; padding-top: 3.6rem; padding-bottom: 5rem;}
h1, h2, h3 {font-family: 'Geist', system-ui, sans-serif !important; letter-spacing: -0.02em;}

/* page header: left aligned, one line of context */
.bt-head {margin: 0 0 4px;}
.bt-head h1 {margin: 0; padding: 0; font-size: 34px; font-weight: 700; line-height: 1.1;}
.bt-head .sub {margin-top: 6px; font-size: 15px; color: #8a8a93;}

/* sections: st.container(key="card_...") */
div[class*="st-key-card_"] {background: #101012; border: 1px solid #222227; border-radius: 16px; padding: 22px 26px 24px; gap: 0.9rem;}
.bt-sec {display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap;}
.bt-sec .t {font-size: 19px; font-weight: 600; letter-spacing: -0.01em;}
.bt-sec .n {font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #6b6b74; margin-right: 10px;}
.bt-sec .h {font-size: 13px; color: #8a8a93;}

/* the one-sentence answer */
.bt-answer {font-size: 22px; line-height: 1.4; font-weight: 500; letter-spacing: -0.01em;}
.bt-answer .m {color: #8a8a93; font-weight: 400;}

/* number tiles */
.bt-stats {display: flex; flex-wrap: wrap; gap: 12px;}
.bt-stat {flex: 1 1 150px; background: #0c0c0e; border: 1px solid #1d1d22; border-radius: 12px; padding: 14px 16px;}
.bt-stat .k {font-size: 13px; color: #8a8a93;}
.bt-stat .v {font-size: 28px; font-weight: 600; letter-spacing: -0.02em; margin-top: 4px;}
.bt-stat .s {font-size: 12px; color: #6b6b74; margin-top: 2px;}

/* label / value rows */
.bt-row {display: flex; flex-wrap: wrap; gap: 4px 24px; padding: 11px 0; border-top: 1px solid #1d1d22;}
.bt-row:first-child {border-top: 0; padding-top: 2px;}
.bt-row .k {flex: 0 0 180px; font-size: 13px; color: #8a8a93; padding-top: 2px;}
.bt-row .v {flex: 1 1 260px; font-size: 15px; line-height: 1.5;}
.bt-muted {color: #8a8a93;}

/* simple tables */
.bt-table {width: 100%; border-collapse: collapse; font-size: 14px;}
.bt-table th {text-align: left; font-weight: 500; font-size: 12px; color: #8a8a93; padding: 0 12px 10px 0; border-bottom: 1px solid #222227; white-space: nowrap;}
.bt-table td {padding: 12px 12px 12px 0; border-bottom: 1px solid #1a1a1f; vertical-align: top;}
.bt-table tr:last-child td {border-bottom: 0;}
.bt-table, .bt-table th, .bt-table td, .bt-table tr {border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; background: transparent !important;}
.bt-table td.num, .bt-table th.num {text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;}
.bt-table .rank {font-family: 'JetBrains Mono', monospace; color: #6b6b74; width: 28px;}
.bt-table .name {font-weight: 600;}
.bt-table .set {color: #8a8a93; font-size: 13px; margin-top: 3px;}
.bt-wrap {overflow-x: auto;}
.bt-pill {display: inline-block; font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 999px; white-space: nowrap;}

/* picker tiles: st.container(key="tile_...") */
div[class*="st-key-tile_"] {background: #0c0c0e; border: 1px solid #1d1d22; border-radius: 12px; padding: 10px 14px 8px; gap: 0;}
div[class*="st-key-tile_"] [data-testid="stCheckbox"] p {font-weight: 600; font-size: 15px;}
div[class*="st-key-tile_f_"] {gap: 0.6rem; padding: 12px 16px 6px;}
.bt-tile-sub {font-size: 12px; color: #6b6b74; margin: -4px 0 2px 28px;}
.bt-brand {display: flex; align-items: center; gap: 10px; font-size: 16px; font-weight: 600; margin: 4px 0 14px;}
.bt-brand img {margin-right: 0 !important;}

/* widgets */
label[data-testid="stWidgetLabel"] p {font-size: 13px !important; color: #8a8a93;}
.stButton button {font-weight: 600; padding: 0.5rem 1.4rem;}
[data-testid="stExpander"] details {background: #101012; border: 1px solid #222227; border-radius: 16px;}
[data-testid="stCaptionContainer"] {color: #6b6b74;}
span[data-baseweb="tag"] {background: #1a1a1f !important; color: #f2f2f3 !important;}
header[data-testid="stHeader"] {background: rgba(10,10,11,0.9); backdrop-filter: blur(8px); border-bottom: 1px solid #1d1d22;}
</style>
"""


def _html(s):
    """st.markdown reads $...$ as math; show dollar signs as plain text."""
    st.markdown(str(s).replace("$", "&#36;"), unsafe_allow_html=True)


def esc(s):
    return _h.escape(str(s))


def style():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, sub="", pills=()):
    _html(f'<div class="bt-head"><h1>{title}</h1>' + (f'<div class="sub">{sub}</div>' if sub else "") + "</div>")


def section(title, hint="", number=""):
    n = f'<span class="n">{number}</span>' if number else ""
    _html(f'<div class="bt-sec"><div class="t">{n}{title}</div><div class="h">{hint}</div></div>')


@contextmanager
def card(title, hint="", color=None, key=None, number=""):
    """A section box. Everything inside the `with` goes in it."""
    with st.container(key=f"card_{key or title.lower().replace(' ', '_')}"):
        section(title, hint, number)
        yield


def pill(verdict):
    c = VERDICT_COLORS.get(verdict, MUTED)
    return f'<span class="bt-pill" style="color:{c}; background:{c}1f">{VERDICT_WORDS.get(verdict, verdict)}</span>'


def stats(items):
    """items: [(label, value, small_text, color or None)] shown as tiles."""
    tiles = "".join(
        f'<div class="bt-stat"><div class="k">{k}</div><div class="v" style="color:{c or TEXT}">{v}</div>'
        + (f'<div class="s">{s}</div>' if s else "")
        + "</div>"
        for k, v, s, c in items
    )
    _html(f'<div class="bt-stats">{tiles}</div>')


def rows(items):
    """items: [(label, html)] as label / value rows."""
    _html("".join(f'<div class="bt-row"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in items))


def table(headers, body, num_cols=()):
    """A plain HTML table. headers: list of str; body: list of rows (lists of html); num_cols: right-aligned columns."""
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(headers))
    tr = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{v}</td>' for i, v in enumerate(r)) + "</tr>"
        for r in body
    )
    _html(f'<div class="bt-wrap"><table class="bt-table"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>')


def plot_layout(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=24, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist, system-ui, sans-serif", color=MUTED, size=12),
        xaxis=dict(showgrid=False, linecolor=LINE), yaxis=dict(gridcolor=LINE, zeroline=False),
        hoverlabel=dict(bgcolor="#101012", bordercolor="#26262b", font=dict(color=TEXT)),
    )
    return fig


# ---- plain-English settings ----------------------------------------------------
_WORDS = {
    "hours": {"ny": "NY session", "london_ny": "London + NY", "all": "all hours", "open": "first 2 hours",
              "ny_open": "NY open (9:30-11:30)", "ny+asia": "NY + Asia", "asia": "Asia (8 PM-2 AM)"},
    "stop": {"swing": "swing stop", "body": "body stop"},
    "setup": {"erl": "ERL (swing sweep)", "irl": "IRL (FVG)"},
    "side": {"both": "longs + shorts", "long": "longs only", "short": "shorts only", "trend": "with the SMA(50) trend"},
}


def _word(k, v):
    if k in _WORDS and v in _WORDS[k]:
        return _WORDS[k][v]
    if k == "smt":
        return "no SMT" if v in ("none", False) else ("SMT" if v is True else f"SMT {str(v).upper()}")
    if k == "psp":
        return "PSP" if v else "no PSP"
    if k == "target":
        return "target C1 range" if v == "range" else f"{v} target"
    if k == "lookback":
        return f"last {v}h"
    if k == "max_wick":
        return "any sweep size" if v >= 1 else f"sweep ≤ {v:.0%} of C1"
    if k == "first_hour":
        return "first hour only" if v else "first 3 hours"
    if k == "target_r":
        return "no take profit" if not v else f"take profit {v}R"
    if k == "stop_atr":
        return f"stop {v} ATR"
    if k == "hold_x":
        return "normal hold" if v == 1 else f"hold {v}x longer"
    if k == "tf":
        return f"{v}-min chart"
    return f"{k.replace('_', ' ')} {v}"


def settings_text(params: dict) -> str:
    return " · ".join(_word(k, v) for k, v in params.items())


# ---- prop firm brands + logos --------------------------------------------------
# Logos load from each firm's own website icon (needs internet; falls back to the name only).
BRANDS = {
    "fundednext": ("FundedNext", "fundednext.com"),
    "topstep": ("Topstep", "topstep.com"),
    "lucid": ("Lucid Trading", "lucidtrading.com"),
    "mffu": ("MyFundedFutures", "myfundedfutures.com"),
    "tradeify": ("Tradeify", "tradeify.co"),
    "apex": ("Apex Trader Funding", "apextraderfunding.com"),
}


def brand(key: str) -> str:
    return key.split("_")[0]


def logo_url(key: str, size=64) -> str:
    dom = BRANDS.get(brand(key), ("", ""))[1]
    return f"https://www.google.com/s2/favicons?domain={dom}&sz={size}" if dom else ""


def logo(key: str, px=20) -> str:
    u = logo_url(key)
    return (f'<img src="{u}" width="{px}" height="{px}" style="border-radius:5px;vertical-align:-4px;margin-right:8px;'
            f'background:#fff">') if u else ""


def account_label(rules) -> str:
    """'Rapid Pro 25K · Eval' style short label, without the brand name."""
    name = rules.name
    for b, _ in BRANDS.values():
        name = name.replace(b, "")
    name = name.replace("MyFundedFutures", "").replace("Lucid", "").replace("Apex", "").replace("Funded (XFA)", "").replace("Funded", "")
    return " ".join(name.split()) or rules.name
