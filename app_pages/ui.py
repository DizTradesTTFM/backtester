"""Shared look for every page: the BmT Journal style (dark cards with window dots, mono labels, colored pills)."""

from contextlib import contextmanager

import streamlit as st

BLUE, AMBER, PINK, GREEN = "#4aa3ff", "#ffb020", "#ff5c8a", "#3ddc84"
TEXT, MUTED, DIM, LINE = "#f2f2f3", "#8a8a93", "#7c7c86", "#202024"
VERDICT_COLORS = {"PASS": GREEN, "WATCH": AMBER, "FAIL": PINK}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.block-container {max-width: 1360px; padding-top: 3.2rem; padding-bottom: 4rem;}
h1, h2, h3 {font-family: 'Geist', system-ui, sans-serif !important; letter-spacing: -0.03em;}
.mono {font-family: 'JetBrains Mono', monospace;}

/* page header */
.bt-head {display: flex; flex-direction: column; align-items: center; gap: 12px; margin: 0 0 18px;}
.bt-head h1 {margin: 0; padding: 0; font-size: 52px; font-weight: 700; letter-spacing: -0.035em; line-height: 1.05; text-align: center;}
.bt-head .sub {font-family: 'JetBrains Mono', monospace; font-size: 14px; letter-spacing: 0.08em; color: #f2f2f3; text-align: center;}
.bt-pills {display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-top: 6px;}
.bt-pill {font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;
          padding: 9px 15px; border: 1px solid #26262b; border-radius: 999px;}

/* cards: st.container(key="card_...") */
div[class*="st-key-card_"] {background: #101012; border: 1px solid #26262b; border-radius: 14px; padding: 0 24px 22px; overflow: hidden; gap: 0.9rem;}
.bt-bar {display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px 16px;
         margin: 0 -24px 6px; padding: 15px 24px; border-bottom: 1px solid #202024; background: #0e0e10;}
.bt-bar .l {display: flex; align-items: center; gap: 16px;}
.bt-dots {display: flex; gap: 8px;} .bt-dots i {width: 12px; height: 12px; border-radius: 50%; display: block;}
.bt-label {font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 0.16em; color: #8a8a93; text-transform: uppercase;}
.bt-tag {font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase;}

/* stat tiles */
.bt-stats {display: flex; flex-wrap: wrap; gap: 14px;}
.bt-stat {flex: 1 1 160px; background: #0e0e10; border: 1px solid #202024; border-radius: 12px; padding: 14px 18px;}
.bt-stat .k {font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.14em; color: #8a8a93; text-transform: uppercase;}
.bt-stat .v {font-size: 30px; font-weight: 600; letter-spacing: -0.02em; margin-top: 6px;}
.bt-stat .s {font-size: 13px; color: #7c7c86; margin-top: 2px;}

/* key/value rows like the journal */
.bt-row {display: flex; flex-wrap: wrap; gap: 6px 24px; padding: 12px 0; border-top: 1px solid #202024;}
.bt-row:first-child {border-top: 0;}
.bt-row .k {flex: 0 0 170px; font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 0.14em; color: #8a8a93; text-transform: uppercase; padding-top: 3px;}
.bt-row .v {flex: 1 1 220px; font-size: 16px; line-height: 1.45;}
.bt-muted {color: #7c7c86; font-size: 14px; line-height: 1.5;}

/* widgets */
label[data-testid="stWidgetLabel"] p {font-family: 'JetBrains Mono', monospace; font-size: 11px !important; letter-spacing: 0.14em; text-transform: uppercase; color: #8a8a93;}
[data-testid="stMetricLabel"] p {font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: #8a8a93;}
.stButton button, .stDownloadButton button {font-family: 'JetBrains Mono', monospace; letter-spacing: 0.1em; text-transform: uppercase; font-size: 12px; padding: 0.55rem 1.3rem;}
[data-testid="stExpander"] details {background: #101012; border: 1px solid #26262b; border-radius: 14px;}
[data-testid="stCaptionContainer"] {color: #7c7c86;}

span[data-baseweb="tag"] {background: #0e0e10 !important; border: 1px solid #26262b; color: #4aa3ff !important;
                          font-family: 'JetBrains Mono', monospace; font-size: 12px;}
span[data-baseweb="tag"] svg {color: #8a8a93;}

/* top navigation as pills */
header[data-testid="stHeader"] {background: rgba(10,10,11,0.85); backdrop-filter: blur(8px); border-bottom: 1px solid #202024;}
header [data-testid="stTopNavLink"], header a[href] {font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;}
</style>
"""


def _html(s):
    """st.markdown reads $...$ as math; show dollar signs as plain text."""
    st.markdown(str(s).replace("$", "&#36;"), unsafe_allow_html=True)


def style():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, sub, pills=()):
    """Big centered title like the journal. pills: [(text, color)]."""
    p = "".join(f'<span class="bt-pill" style="color:{c}">{t}</span>' for t, c in pills)
    st.markdown(
        f'<div class="bt-head"><h1>{title}</h1><div class="sub">{sub}</div>'
        + (f'<div class="bt-pills">{p}</div>' if p else "")
        + "</div>",
        unsafe_allow_html=True,
    )


@contextmanager
def card(label, tag="", color=BLUE, key=None):
    """A journal-style window card. Everything inside the `with` goes in the card."""
    with st.container(key=f"card_{key or label.lower().replace(' ', '_')}"):
        st.markdown(
            '<div class="bt-bar"><div class="l"><div class="bt-dots"><i style="background:#ff5f57"></i>'
            '<i style="background:#febc2e"></i><i style="background:#28c840"></i></div>'
            f'<div class="bt-label">{label}</div></div><div class="bt-tag" style="color:{color}">{tag}</div></div>',
            unsafe_allow_html=True,
        )
        yield


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
    """items: [(label, html)] as journal key/value rows."""
    _html("".join(f'<div class="bt-row"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in items))


def plot_layout(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist, system-ui, sans-serif", color=MUTED, size=12),
        xaxis=dict(showgrid=False, linecolor=LINE), yaxis=dict(gridcolor=LINE, zeroline=False),
        hoverlabel=dict(bgcolor="#101012", bordercolor="#26262b", font=dict(color=TEXT, family="JetBrains Mono")),
    )
    return fig
