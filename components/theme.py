"""
components/theme.py
-------------------
Design system for SPIN-SENTRY main application (port 8501).

Used exclusively by:
  pages/1_Home.py
  pages/2_Research_Workspace.py
  pages/3_Analysis_Results.py
  pages/4_Dataset_Model_Lab.py
  pages/5_Methodology.py

NOT used by scripts/review_claim_annotations.py (port 8503 annotation app).
Safe to modify without affecting the annotation app.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens — aligned with specification
# ---------------------------------------------------------------------------

COLORS = {
    # Backgrounds
    "bg_primary":    "#0B1220",
    "bg_secondary":  "#111B2E",
    "bg_card":       "#162238",
    "bg_card_hover": "#1A2A42",
    # Borders
    "border":        "#29374D",
    "border_light":  "#344862",
    # Text
    "text_primary":  "#F3F6FC",
    "text_secondary": "#A8B5C8",
    "text_muted":    "#64748B",
    # Accents
    "blue":          "#3B82F6",
    "cyan":          "#22D3EE",
    "violet":        "#8B5CF6",
    # Status
    "green":         "#22C55E",
    "amber":         "#F59E0B",
    "red":           "#EF4444",
    "teal":          "#14B8A6",
}


# ---------------------------------------------------------------------------
# CSS injection — main app only
# ---------------------------------------------------------------------------

def inject_custom_css():
    """Inject the professional dark theme CSS for the main app.

    Called once at the top of every main-app page.
    Does NOT affect scripts/review_claim_annotations.py.
    """
    st.markdown(f"""
    <style>
    /* ── Import font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Root ── */
    :root {{
        --bg-primary:   {COLORS["bg_primary"]};
        --bg-secondary: {COLORS["bg_secondary"]};
        --bg-card:      {COLORS["bg_card"]};
        --border:       {COLORS["border"]};
        --text-primary: {COLORS["text_primary"]};
        --text-secondary: {COLORS["text_secondary"]};
        --text-muted:   {COLORS["text_muted"]};
        --blue:         {COLORS["blue"]};
        --cyan:         {COLORS["cyan"]};
        --green:        {COLORS["green"]};
        --amber:        {COLORS["amber"]};
        --red:          {COLORS["red"]};
    }}

    /* ── Global ── */
    .stApp {{
        background: {COLORS["bg_primary"]};
        color: {COLORS["text_primary"]};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {COLORS["bg_secondary"]};
        border-right: 1px solid {COLORS["border"]};
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {{
        color: {COLORS["text_secondary"]};
        font-size: 0.88rem;
    }}
    section[data-testid="stSidebar"] label {{
        color: {COLORS["text_secondary"]} !important;
        font-size: 0.85rem !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
        background: {COLORS["bg_card"]} !important;
        border-color: {COLORS["border"]} !important;
        color: {COLORS["text_primary"]} !important;
    }}

    /* ── Typography ── */
    h1 {{
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: {COLORS["text_primary"]} !important;
        letter-spacing: -0.3px;
        margin-bottom: 0.25rem !important;
    }}
    h2 {{
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: {COLORS["text_secondary"]} !important;
        border-bottom: 1px solid {COLORS["border"]};
        padding-bottom: 0.5rem;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }}
    h3 {{
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: {COLORS["text_primary"]} !important;
    }}
    p {{
        color: {COLORS["text_secondary"]};
        line-height: 1.65;
    }}
    code {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 4px;
        padding: 0.1rem 0.4rem;
        font-size: 0.85em;
        color: {COLORS["cyan"]};
    }}

    /* ── Page title area ── */
    .ss-page-header {{
        padding: 0 0 1.25rem 0;
        border-bottom: 1px solid {COLORS["border"]};
        margin-bottom: 2rem;
    }}
    .ss-page-title {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {COLORS["text_primary"]};
        letter-spacing: -0.3px;
        margin: 0;
        line-height: 1.3;
    }}
    .ss-page-description {{
        font-size: 0.95rem;
        color: {COLORS["text_secondary"]};
        margin-top: 0.4rem;
        line-height: 1.55;
    }}

    /* ── Section heading ── */
    .ss-section-heading {{
        font-size: 0.78rem;
        font-weight: 700;
        color: {COLORS["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 0.9px;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid {COLORS["border"]};
    }}

    /* ── Cards ── */
    .spin-card {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 0.875rem;
    }}
    .spin-card h4 {{
        color: {COLORS["text_primary"]} !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin: 0 0 0.5rem 0 !important;
    }}
    .spin-card p {{
        color: {COLORS["text_secondary"]};
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
        line-height: 1.55;
    }}

    /* ── Feature cards ── */
    .feature-card {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 1.5rem 1.25rem;
        height: 100%;
    }}
    .feature-card:hover {{
        border-color: {COLORS["border_light"]};
    }}
    .feature-icon {{
        font-size: 1.6rem;
        margin-bottom: 0.75rem;
        display: block;
    }}
    .feature-card h4 {{
        color: {COLORS["text_primary"]} !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }}
    .feature-card p {{
        color: {COLORS["text_secondary"]};
        font-size: 0.87rem;
        line-height: 1.55;
        margin: 0;
    }}

    /* ── Hero ── */
    .hero-section {{
        padding: 1.5rem 0 2rem;
    }}
    .hero-eyebrow {{
        font-size: 0.72rem;
        font-weight: 700;
        color: {COLORS["blue"]};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.75rem;
        display: block;
    }}
    .hero-title {{
        font-size: 2.4rem;
        font-weight: 800;
        color: {COLORS["text_primary"]};
        letter-spacing: -0.5px;
        line-height: 1.15;
        margin-bottom: 0.75rem;
    }}
    .hero-title span {{
        color: {COLORS["blue"]};
    }}
    .hero-subtitle {{
        font-size: 1rem;
        color: {COLORS["text_secondary"]};
        max-width: 680px;
        line-height: 1.65;
        margin-bottom: 0;
    }}

    /* ── Stat cards ── */
    .stat-card {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 1rem 1.1rem;
        text-align: center;
    }}
    .stat-value {{
        font-size: 1.75rem;
        font-weight: 800;
        color: {COLORS["text_primary"]};
        line-height: 1.2;
        letter-spacing: -0.3px;
    }}
    .stat-label {{
        font-size: 0.72rem;
        font-weight: 600;
        color: {COLORS["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 0.3rem;
    }}
    .stat-accent-blue   .stat-value {{ color: {COLORS["blue"]}; }}
    .stat-accent-cyan   .stat-value {{ color: {COLORS["cyan"]}; }}
    .stat-accent-green  .stat-value {{ color: {COLORS["green"]}; }}
    .stat-accent-amber  .stat-value {{ color: {COLORS["amber"]}; }}
    .stat-accent-red    .stat-value {{ color: {COLORS["red"]}; }}

    /* ── Architecture steps ── */
    .arch-step {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-left: 3px solid {COLORS["blue"]};
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }}
    .arch-step-num {{
        display: inline-block;
        background: {COLORS["blue"]}18;
        color: {COLORS["blue"]};
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.5rem;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }}
    .arch-step h5 {{
        display: inline;
        color: {COLORS["text_primary"]} !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }}
    .arch-step p {{
        color: {COLORS["text_secondary"]};
        font-size: 0.87rem;
        margin: 0.5rem 0 0 0;
        line-height: 1.55;
    }}

    /* ── Status badges ── */
    .badge {{
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }}
    .badge-supported {{
        background: {COLORS["green"]}18;
        color: {COLORS["green"]};
        border: 1px solid {COLORS["green"]}35;
    }}
    .badge-flagged {{
        background: {COLORS["red"]}18;
        color: {COLORS["red"]};
        border: 1px solid {COLORS["red"]}35;
    }}
    .badge-weak {{
        background: {COLORS["amber"]}18;
        color: {COLORS["amber"]};
        border: 1px solid {COLORS["amber"]}35;
    }}
    .badge-protocol {{
        background: {COLORS["blue"]}18;
        color: {COLORS["blue"]};
        border: 1px solid {COLORS["blue"]}35;
    }}
    .badge-neutral {{
        background: {COLORS["text_muted"]}18;
        color: {COLORS["text_muted"]};
        border: 1px solid {COLORS["text_muted"]}35;
    }}

    /* ── Risk level pill ── */
    .risk-low    {{ color: {COLORS["green"]}; font-weight: 700; }}
    .risk-mod    {{ color: {COLORS["amber"]}; font-weight: 700; }}
    .risk-high   {{ color: {COLORS["red"]};   font-weight: 700; }}

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {{
        border: 1px dashed {COLORS["border_light"]};
        border-radius: 8px;
        padding: 0.5rem;
        background: {COLORS["bg_card"]};
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: {COLORS["blue"]}60;
    }}

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {{
        background: {COLORS["blue"]} !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.4rem !important;
        letter-spacing: 0.1px;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: #2563EB !important;
        box-shadow: 0 2px 12px {COLORS["blue"]}40 !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: {COLORS["bg_card"]} !important;
        color: {COLORS["text_primary"]} !important;
        border: 1px solid {COLORS["border_light"]} !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: {COLORS["blue"]}60 !important;
        color: {COLORS["blue"]} !important;
    }}
    .stButton > button:disabled {{
        opacity: 0.4 !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {COLORS["bg_secondary"]};
        border-bottom: 1px solid {COLORS["border"]};
        gap: 0;
        padding: 0;
        border-radius: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 0 !important;
        color: {COLORS["text_muted"]} !important;
        font-weight: 500 !important;
        font-size: 0.87rem !important;
        padding: 0.65rem 1.2rem !important;
        border-bottom: 2px solid transparent !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: transparent !important;
        color: {COLORS["blue"]} !important;
        border-bottom-color: {COLORS["blue"]} !important;
        font-weight: 600 !important;
    }}

    /* ── Dataframe / tables ── */
    .stDataFrame {{
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        overflow: hidden;
    }}
    .stDataFrame thead tr th {{
        background: {COLORS["bg_secondary"]} !important;
        color: {COLORS["text_muted"]} !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .stDataFrame tbody tr td {{
        font-size: 0.88rem !important;
        color: {COLORS["text_secondary"]} !important;
    }}

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS["text_muted"]} !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* ── Expanders ── */
    details {{
        background: {COLORS["bg_card"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
    }}
    details summary {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }}

    /* ── Info / warning / error ── */
    .stAlert {{
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }}

    /* ── Progress bar ── */
    .stProgress > div > div {{
        background: {COLORS["blue"]} !important;
        border-radius: 4px;
    }}
    .stProgress > div {{
        background: {COLORS["border"]} !important;
        border-radius: 4px;
    }}

    /* ── Dividers ── */
    hr {{
        border-color: {COLORS["border"]} !important;
        margin: 1.5rem 0 !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {COLORS["bg_primary"]}; }}
    ::-webkit-scrollbar-thumb {{ background: {COLORS["border"]}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {COLORS["border_light"]}; }}

    /* ── Evidence item ── */
    .evidence-item {{
        background: {COLORS["bg_secondary"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: {COLORS["text_secondary"]};
        line-height: 1.55;
    }}
    .evidence-score {{
        font-size: 0.78rem;
        color: {COLORS["text_muted"]};
        margin-top: 0.35rem;
    }}
    .evidence-claim {{
        font-size: 0.92rem;
        font-weight: 500;
        color: {COLORS["text_primary"]};
        line-height: 1.55;
        margin-bottom: 0.6rem;
    }}

    /* ── Disclaimer banner ── */
    .ss-disclaimer {{
        background: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-left: 3px solid {COLORS["amber"]};
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: {COLORS["text_muted"]};
        line-height: 1.55;
    }}

    /* ── Inline score pill ── */
    .score-pill {{
        display: inline-block;
        background: {COLORS["bg_secondary"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 4px;
        padding: 0.1rem 0.5rem;
        font-size: 0.78rem;
        font-family: 'SFMono-Regular', Consolas, monospace;
        color: {COLORS["text_secondary"]};
    }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable component helpers
# ---------------------------------------------------------------------------

def render_page_header(title: str, description: str = ""):
    """Render a consistent page header with title and optional description."""
    desc_html = (
        f'<p class="ss-page-description">{description}</p>' if description else ""
    )
    st.markdown(
        f'<div class="ss-page-header">'
        f'<h1 class="ss-page-title">{title}</h1>'
        f'{desc_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_section_heading(title: str):
    """Render a small-caps section heading."""
    st.markdown(
        f'<div class="ss-section-heading">{title}</div>',
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, badge: str = ""):
    """Render a left-aligned hero section (kept for compatibility)."""
    eyebrow = f'<span class="hero-eyebrow">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="hero-section">'
        f'{eyebrow}'
        f'<div class="hero-title">{title}</div>'
        f'<p class="hero-subtitle">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_card(title: str, content: str, icon: str = ""):
    """Render a styled content card."""
    icon_html = f'<span style="margin-right:0.4rem;">{icon}</span>' if icon else ""
    st.markdown(
        f'<div class="spin-card">'
        f'<h4>{icon_html}{title}</h4>'
        f'<p>{content}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_feature_card(icon: str, title: str, description: str):
    """Render a feature card."""
    st.markdown(
        f'<div class="feature-card">'
        f'<span class="feature-icon">{icon}</span>'
        f'<h4>{title}</h4>'
        f'<p>{description}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_stat_card(value: str, label: str, color: str = ""):
    """Render a KPI stat card. color overrides stat-value color."""
    style = f'style="color:{color};"' if color else ""
    st.markdown(
        f'<div class="stat-card">'
        f'<div class="stat-value" {style}>{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_arch_step(number: int, title: str, description: str, accent: str = ""):
    """Render an architecture pipeline step."""
    border_color = accent or COLORS["blue"]
    bg = f"{border_color}18"
    st.markdown(
        f'<div class="arch-step" style="border-left-color:{border_color};">'
        f'<span class="arch-step-num" style="background:{bg};color:{border_color};">'
        f'Stage {number}</span>'
        f'<h5>{title}</h5>'
        f'<p>{description}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_badge(text: str, variant: str = "protocol") -> str:
    """Return HTML for an inline badge.

    variant: supported | flagged | weak | protocol | neutral
    """
    return f'<span class="badge badge-{variant}">{text}</span>'


def render_disclaimer(text: str):
    """Render a left-accented disclaimer banner."""
    st.markdown(
        f'<div class="ss-disclaimer">{text}</div>',
        unsafe_allow_html=True,
    )


def render_risk_banner(risk_label: str, flagged: int, supported: int, avg_we: float):
    """Render a prominent risk-level summary banner near the top of results.

    Shows the overall spin-risk level with color coding and a brief
    interpretation note. Always includes the screening disclaimer.
    """
    if "Low" in risk_label:
        bg      = f"{COLORS['green']}15"
        border  = COLORS["green"]
        color   = COLORS["green"]
        interp  = "No contradictions detected. All empirical claims had supporting evidence."
    elif "Moderate" in risk_label:
        bg      = f"{COLORS['amber']}15"
        border  = COLORS["amber"]
        color   = COLORS["amber"]
        interp  = "One or more empirical claims had weak or absent supporting evidence."
    else:
        bg      = f"{COLORS['red']}15"
        border  = COLORS["red"]
        color   = COLORS["red"]
        interp  = "At least one direct contradiction between abstract claim and body text was detected."

    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {border}40;border-left:4px solid {border};
             border-radius:0 8px 8px 0;padding:1rem 1.25rem;margin-bottom:1.25rem;">
          <div style="display:flex;align-items:baseline;gap:0.75rem;flex-wrap:wrap;">
            <span style="font-size:0.72rem;font-weight:700;color:{COLORS['text_muted']};
                  text-transform:uppercase;letter-spacing:0.8px;">Overall Spin Risk</span>
            <span style="font-size:1.4rem;font-weight:800;color:{color};letter-spacing:-0.3px;">
              {risk_label}
            </span>
            <span style="font-size:0.82rem;color:{COLORS['text_secondary']};">
              &nbsp;·&nbsp; {flagged} flagged &nbsp;/&nbsp; {supported} supported &nbsp;·&nbsp;
              avg WE {avg_we:.3f}
            </span>
          </div>
          <div style="font-size:0.82rem;color:{COLORS['text_secondary']};margin-top:0.35rem;">
            {interp}
          </div>
          <div style="font-size:0.76rem;color:{COLORS['text_muted']};margin-top:0.5rem;
               border-top:1px solid {border}25;padding-top:0.4rem;">
            Automated risk levels are heuristic screening signals for human review.
            They are not clinical diagnoses or proof of scientific misconduct.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav():
    """Render consistent sidebar branding across all main-app pages."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 1rem 0 0.75rem;">
                <div style="font-size: 1.1rem; font-weight: 800;
                     color: {COLORS["text_primary"]}; letter-spacing: -0.3px;">
                    SPIN-SENTRY
                </div>
                <div style="font-size: 0.72rem; font-weight: 600;
                     color: {COLORS["text_muted"]}; text-transform: uppercase;
                     letter-spacing: 0.8px; margin-top: 0.2rem;">
                    Research Intelligence Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
