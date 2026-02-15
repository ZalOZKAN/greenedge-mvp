"""GreenEdge-5G  Streamlit Dashboard.

Run:
    streamlit run greenedge/dashboard/app.py
"""

from __future__ import annotations

import io
import sys
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# --- Ensure repo root is on sys.path so 'greenedge' package is importable ---
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from greenedge.rl.baselines import greedy_min_energy, greedy_min_latency, simple_threshold
from greenedge.simulator.config import EnvConfig
from greenedge.simulator.env import ACTION_LABELS, GreenEdgeEnv

# ---------------------------------------------------------------------------
# POLICY_MAP — Tek kaynak, her yerde bu kullanılacak
# ---------------------------------------------------------------------------
POLICY_MAP = {
    "rl_ppo": {"tr": "YZ", "en": "AI"},
    "greedy_min_latency": {"tr": "Hız", "en": "Speed"},
    "greedy_min_energy": {"tr": "Maliyet", "en": "Cost"},
    "simple_threshold": {"tr": "CPU", "en": "CPU"},
}

# Internal key list (order matters for colors)
POLICY_KEYS = ["rl_ppo", "greedy_min_latency", "greedy_min_energy", "simple_threshold"]
POLICY_COLORS = ["#0d6efd", "#198754", "#fd7e14", "#dc3545"]

# ---------------------------------------------------------------------------
# i18n — Türkçe (varsayılan) / English
# ---------------------------------------------------------------------------
TEXTS = {
    "tr": {
        "page_title": "GreenEdge-5G Kontrol Paneli",
        "title": "GreenEdge-5G",
        "subtitle": "5G kenar/bulut altyapısı için yapay zeka destekli iş yükü yönlendirme",
        "download_pdf": "Raporu İndir (PDF)",
        "deploy_btn": "Uygula",
        "lang_label": "Dil / Language",
        "eval_header": "Değerlendirme Özeti",
        "eval_info_title": "Açıklama",
        "eval_info_content": "Yapay zeka modelimizin 200 test senaryosu üzerindeki performansı. Amaç: en düşük gecikme ve en az enerji tüketimi ile iş yüklerini yönlendirmek, SLA ihlallerini sıfıra yakın tutmak.",
        "terms_explain": "**Terimler:** **YZ** = Yapay Zeka (Pekiştirmeli Öğrenme modeli) · **P95** = 100 istekten 95'inin bu sürede tamamlandığı gecikme · **SLA** = Hizmet Seviyesi Anlaşması (maks. 120 ms).",
        "avg_latency": "Ort. Gecikme",
        "p95_latency": "P95 Gecikme",
        "energy_mbps": "Enerji/Mbps",
        "sla_viol": "SLA İhlal",
        "target_label": "Hedef: {val}",
        "comparison": "Politika Karşılaştırması",
        "col_policy": "Politika",
        "col_avg_reward": "Ödül",
        "col_avg_lat": "Gecikme",
        "col_p95_lat": "P95",
        "col_energy": "Enerji",
        "col_sla": "SLA %",
        "winner_badge": "🏆 Kazanan",
        "tradeoff_title": "Gecikme – Enerji Dengesi",
        "tradeoff_x": "Enerji / Mbps",
        "tradeoff_y": "Gecikme (ms)",
        "tradeoff_explain": "İdeal konum sol-alt köşedir (düşük enerji + düşük gecikme).",
        "live_header": "Canlı Simülasyon",
        "live_intro": "Seçtiğiniz politika ile 50 adımlık simülasyon. Mavi=Gecikme (sol), Turuncu=Enerji (sağ).",
        "policy_label": "Politika",
        "seed_label": "Senaryo No",
        "run_btn": "Başlat",
        "live_chart_title": "Adım Adım: {policy}",
        "live_x": "Adım",
        "live_y_lat": "Gecikme (ms)",
        "live_y_eng": "Enerji",
        "last_n": "Son 10 Karar",
        "last_n_explain": "Son 10 yönlendirme kararı: hedef, gecikme, enerji, SLA durumu.",
        "col_step": "Adım",
        "col_target": "Hedef",
        "col_lat": "Gecikme",
        "col_eng": "Enerji",
        "col_sla_step": "SLA",
        "col_reward": "Ödül",
        "sla_ok": "✓",
        "sla_fail": "✗",
        "ep_done_title": "SİMÜLASYON TAMAMLANDI",
        "ep_done_detail": "Ödül: {reward:.2f} · Gecikme: {lat:.1f} ms · SLA İhlal: {sla:.1f}%",
        "cmp_header": "Politika Karşılaştırması",
        "cmp_intro": "Aynı senaryoda tüm politikaları karşılaştırın. Yükselen eğri = daha iyi.",
        "cmp_seed": "Senaryo No",
        "cmp_btn": "Karşılaştır",
        "cmp_title": "Kümülatif Ödül",
        "cmp_x": "Adım",
        "cmp_y": "Kümülatif Ödül",
        "no_results": "Sonuç dosyası bulunamadı. Önce: `python -m greenedge.rl.evaluate`",
        "pdf_title": "GreenEdge-5G Değerlendirme Raporu",
        "pdf_generated": "Oluşturulma Tarihi",
        "pdf_scenarios": "Test Senaryosu Sayısı",
        "pdf_kpi_section": "KPI Özeti (PPO Modeli)",
        "pdf_comparison_section": "Politika Karşılaştırması",
        "pdf_winner_section": "Kazanan Politika",
        "pdf_footer": "GreenEdge-5G MVP · github.com/greenedge/mvp",
        "realtime_header": "Gerçek Zamanlı Demo",
        "realtime_intro": "Otomatik yenilenen canlı simülasyon. Her 2 saniyede yeni bir karar alınır.",
        "realtime_start": "Başlat",
        "realtime_stop": "Durdur",
        "realtime_status": "Durum",
        "realtime_running": "🟢 Çalışıyor",
        "realtime_stopped": "🔴 Durduruldu",
        "ab_header": "A/B Test Paneli",
        "ab_intro": "İki politikayı aynı senaryoda yan yana karşılaştırın.",
        "ab_policy_a": "Politika A",
        "ab_policy_b": "Politika B",
        "ab_run": "Karşılaştır",
        "ab_winner": "Kazanan: {policy}",
        "ab_metric_lat": "Gecikme",
        "ab_metric_energy": "Enerji",
        "ab_metric_sla": "SLA",
        "ab_better": "↓ daha iyi",
    },
    "en": {
        "page_title": "GreenEdge-5G Dashboard",
        "title": "GreenEdge-5G",
        "subtitle": "AI-powered workload routing for 5G edge / cloud infrastructure",
        "download_pdf": "Download Report (PDF)",
        "deploy_btn": "Deploy",
        "lang_label": "Dil / Language",
        "eval_header": "Evaluation Summary",
        "eval_info_title": "Info",
        "eval_info_content": "AI model performance across 200 test scenarios. Goal: route workloads with lowest latency and minimal energy, keeping SLA violations near zero.",
        "terms_explain": "**Terms:** **AI** = Artificial Intelligence (Reinforcement Learning model) · **P95** = 95th percentile latency (95% of requests complete within this time) · **SLA** = Service Level Agreement (max 120 ms).",
        "avg_latency": "Avg Latency",
        "p95_latency": "P95 Latency",
        "energy_mbps": "Energy/Mbps",
        "sla_viol": "SLA Viol.",
        "target_label": "Target: {val}",
        "comparison": "Policy Comparison",
        "col_policy": "Policy",
        "col_avg_reward": "Reward",
        "col_avg_lat": "Latency",
        "col_p95_lat": "P95",
        "col_energy": "Energy",
        "col_sla": "SLA %",
        "winner_badge": "🏆 Winner",
        "tradeoff_title": "Latency – Energy Trade-off",
        "tradeoff_x": "Energy / Mbps",
        "tradeoff_y": "Latency (ms)",
        "tradeoff_explain": "Ideal position is bottom-left (low energy + low latency).",
        "live_header": "Live Simulation",
        "live_intro": "Run 50-step simulation. Blue=Latency (left), Orange=Energy (right).",
        "policy_label": "Policy",
        "seed_label": "Scenario #",
        "run_btn": "Run",
        "live_chart_title": "Step-by-Step: {policy}",
        "live_x": "Step",
        "live_y_lat": "Latency (ms)",
        "live_y_eng": "Energy",
        "last_n": "Last 10 Decisions",
        "last_n_explain": "Last 10 routing decisions: target, latency, energy, SLA status.",
        "col_step": "Step",
        "col_target": "Target",
        "col_lat": "Latency",
        "col_eng": "Energy",
        "col_sla_step": "SLA",
        "col_reward": "Reward",
        "sla_ok": "✓",
        "sla_fail": "✗",
        "ep_done_title": "SIMULATION COMPLETE",
        "ep_done_detail": "Reward: {reward:.2f} · Latency: {lat:.1f} ms · SLA Viol: {sla:.1f}%",
        "cmp_header": "Policy Comparison",
        "cmp_intro": "Compare all policies on same scenario. Rising curve = better.",
        "cmp_seed": "Scenario #",
        "cmp_btn": "Compare",
        "cmp_title": "Cumulative Reward",
        "cmp_x": "Step",
        "cmp_y": "Cumulative Reward",
        "no_results": "No results file. Run `python -m greenedge.rl.evaluate` first.",
        "pdf_title": "GreenEdge-5G Evaluation Report",
        "pdf_generated": "Generated",
        "pdf_scenarios": "Test Scenarios",
        "pdf_kpi_section": "KPI Summary (PPO Model)",
        "pdf_comparison_section": "Policy Comparison",
        "pdf_winner_section": "Winning Policy",
        "pdf_footer": "GreenEdge-5G MVP · github.com/greenedge/mvp",
        "realtime_header": "Real-Time Demo",
        "realtime_intro": "Auto-refreshing live simulation. A new decision is made every 2 seconds.",
        "realtime_start": "Start",
        "realtime_stop": "Stop",
        "realtime_status": "Status",
        "realtime_running": "🟢 Running",
        "realtime_stopped": "🔴 Stopped",
        "ab_header": "A/B Test Panel",
        "ab_intro": "Compare two policies side-by-side on the same scenario.",
        "ab_policy_a": "Policy A",
        "ab_policy_b": "Policy B",
        "ab_run": "Compare",
        "ab_winner": "Winner: {policy}",
        "ab_metric_lat": "Latency",
        "ab_metric_energy": "Energy",
        "ab_metric_sla": "SLA",
        "ab_better": "↓ better",
    },
}

TARGET_LABELS = {
    "tr": {"edge-a": "Kenar-A", "edge-b": "Kenar-B", "cloud": "Bulut"},
    "en": {"edge-a": "Edge-A", "edge-b": "Edge-B", "cloud": "Cloud"},
}

# ---------------------------------------------------------------------------
# Typography & Chart Constants
# ---------------------------------------------------------------------------
CHART_FONT = dict(family="Segoe UI, Arial, sans-serif", size=14, color="#212529")
AXIS_FONT_SIZE = 13
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 13

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
RESULTS_JSON = EXPERIMENTS / "results.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plabel(key: str, lang: str) -> str:
    """Get short policy label from POLICY_MAP."""
    return POLICY_MAP.get(key, {}).get(lang, key)


def _tlabel(key: str, lang: str) -> str:
    """Get human-readable target label."""
    return TARGET_LABELS[lang].get(key, key)


@st.cache_data
def load_results() -> Dict:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_sb3_policy():
    if "sb3_model" not in st.session_state:
        from stable_baselines3 import DQN, PPO
        policy_path = EXPERIMENTS / "policy"
        for cls in (PPO, DQN):
            try:
                st.session_state["sb3_model"] = cls.load(str(policy_path))
                break
            except Exception:
                continue
        else:
            st.session_state["sb3_model"] = None
    return st.session_state["sb3_model"]


def run_live_episode(policy_name: str, seed: int = 99) -> List[Dict]:
    cfg = EnvConfig(seed=seed)
    env = GreenEdgeEnv(config=cfg)
    obs, _ = env.reset()
    policy_fn = {
        "rl_ppo": _rl_predict,
        "greedy_min_latency": greedy_min_latency,
        "greedy_min_energy": greedy_min_energy,
        "simple_threshold": simple_threshold,
    }[policy_name]
    steps: List[Dict] = []
    done = False
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        info["reward"] = round(reward, 4)
        steps.append(info)
        done = terminated or truncated
    return steps


def _rl_predict(obs: np.ndarray) -> int:
    model = _load_sb3_policy()
    if model is None:
        return greedy_min_latency(obs)
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


# ---------------------------------------------------------------------------
# PDF Generation
# ---------------------------------------------------------------------------
def generate_pdf(results: Dict, lang: str) -> bytes:
    """Generate PDF report using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    
    T = TEXTS[lang]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=20, textColor=colors.HexColor("#0d6efd"))
    h2_style = ParagraphStyle('CustomH2', parent=styles['Heading2'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#212529"))
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=12, spaceAfter=8)
    
    story = []
    
    # Title
    story.append(Paragraph(T["pdf_title"], title_style))
    story.append(Paragraph(f"{T['pdf_generated']}: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Paragraph(f"{T['pdf_scenarios']}: 200", body_style))
    story.append(Spacer(1, 20))
    
    # KPI Section
    story.append(Paragraph(T["pdf_kpi_section"], h2_style))
    
    ppo_data = results.get("rl_ppo", {})
    if ppo_data:
        kpi_data = [
            [T["avg_latency"], f"{ppo_data.get('avg_latency', 0):.1f} ms"],
            [T["p95_latency"], f"{ppo_data.get('p95_latency', 0):.1f} ms"],
            [T["energy_mbps"], f"{ppo_data.get('avg_energy_per_mbps', 0):.4f}"],
            [T["sla_viol"], f"{ppo_data.get('sla_violation_rate', 0)*100:.1f}%"],
        ]
        kpi_table = Table(kpi_data, colWidths=[8*cm, 6*cm])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f8f9fa")),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ]))
        story.append(kpi_table)
    
    story.append(Spacer(1, 20))
    
    # Comparison Table
    story.append(Paragraph(T["pdf_comparison_section"], h2_style))
    
    header = [T["col_policy"], T["col_avg_reward"], T["col_avg_lat"], T["col_p95_lat"], T["col_energy"], T["col_sla"]]
    table_data = [header]
    
    winner_key = max(results.keys(), key=lambda k: results[k]["avg_reward"]) if results else None
    
    for key in POLICY_KEYS:
        if key in results:
            data = results[key]
            row = [
                _plabel(key, lang),
                f"{data['avg_reward']:.2f}",
                f"{data['avg_latency']:.1f}",
                f"{data['p95_latency']:.1f}",
                f"{data['avg_energy_per_mbps']:.4f}",
                f"{data['sla_violation_rate']*100:.1f}%",
            ]
            table_data.append(row)
    
    comp_table = Table(table_data, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm, 2.5*cm])
    
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]
    
    # Highlight winner row
    if winner_key:
        winner_idx = POLICY_KEYS.index(winner_key) + 1 if winner_key in POLICY_KEYS else 1
        table_style.append(('BACKGROUND', (0, winner_idx), (-1, winner_idx), colors.HexColor("#d4edda")))
    
    comp_table.setStyle(TableStyle(table_style))
    story.append(comp_table)
    
    story.append(Spacer(1, 20))
    
    # Winner Section
    if winner_key:
        story.append(Paragraph(T["pdf_winner_section"], h2_style))
        winner_text = f"🏆 {_plabel(winner_key, lang)}"
        story.append(Paragraph(winner_text, ParagraphStyle('Winner', parent=styles['Normal'], fontSize=18, textColor=colors.HexColor("#155724"))))
    
    story.append(Spacer(1, 40))
    
    # Footer
    story.append(Paragraph(T["pdf_footer"], ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, textColor=colors.gray)))
    
    doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GreenEdge-5G",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Sidebar: language switch
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=48)
    lang = st.radio(
        "Dil / Language",
        options=["tr", "en"],
        format_func=lambda x: "Türkçe" if x == "tr" else "English",
        index=0,
        horizontal=True,
    )

T = TEXTS[lang]

# ---------------------------------------------------------------------------
# Global CSS — Typography Standard
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Hide default Streamlit hamburger & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Typography Scale - Streamlit specific selectors */
    h1, .stMarkdown h1, [data-testid="stMarkdownContainer"] h1 { 
        font-size: 38px !important; font-weight: 700 !important; margin-bottom: 0.5rem !important; 
    }
    h2, .stMarkdown h2, [data-testid="stMarkdownContainer"] h2,
    [data-testid="stHeadingWithArrowContainer"] { 
        font-size: 29px !important; font-weight: 600 !important; 
        border-bottom: 2px solid #0d6efd; padding-bottom: 0.5rem; margin-top: 2rem !important; margin-bottom: 1.2rem !important;
    }
    h3, .stMarkdown h3 {
        margin-bottom: 0.8rem !important;
    }
    h3, .stMarkdown h3, [data-testid="stMarkdownContainer"] h3 { 
        font-size: 22px !important; font-weight: 600 !important; margin-bottom: 0.8rem !important;
    }
    
    /* Body text - exclude headings */
    p, .stMarkdown p { font-size: 14px !important; line-height: 1.5 !important; }
    
    /* Captions / notes */
    .stCaption, small, .element-container small { font-size: 12px !important; color: #6c757d !important; }
    
    /* KPI cards */
    [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 13px !important; font-weight: 500 !important; }
    [data-testid="stMetricDelta"] { font-size: 11px !important; }
    
    /* Tables */
    .stDataFrame td, .stDataFrame th { font-size: 13px !important; padding: 8px !important; }
    
    /* Buttons */
    .stButton button, .stDownloadButton button { 
        font-size: 12px !important;
        padding: 0.4rem 0.8rem !important;
        min-height: 36px !important;
    }
    
    /* Info boxes */
    .stAlert p { font-size: 13px !important; }
    
    /* Header bar */
    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e9ecef;
    }
    .header-left h1 { margin: 0 !important; }
    .header-left p { margin: 0 !important; color: #6c757d; font-size: 13px !important; }
    
    /* Winner badge */
    .winner-badge {
        display: inline-block;
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
    
    /* Consistent card heights & equal widths */
    [data-testid="stMetric"] {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    [data-testid="column"] {
        min-width: 0 !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 1 1 0 !important;
        width: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
results = load_results()

# ---------------------------------------------------------------------------
# Header Bar
# ---------------------------------------------------------------------------
col_title, col_spacer, col_buttons = st.columns([5, 2, 2])

with col_title:
    st.markdown(f"# {T['title']}")
    st.caption(T["subtitle"])

with col_buttons:
    btn_cols = st.columns(2)
    
    # PDF download using ReportLab
    if results:
        pdf_bytes = generate_pdf(results, lang)
        btn_cols[0].download_button(
            label=T['download_pdf'],
            data=pdf_bytes,
            file_name=f"greenedge_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )
    else:
        btn_cols[0].button(T['download_pdf'], disabled=True)
    
    btn_cols[1].button(T['deploy_btn'], disabled=True, key="deploy_btn_header")

# ---------------------------------------------------------------------------
# 1) KPI Cards — Evaluation Summary
# ---------------------------------------------------------------------------
if results:
    st.header(f":material/analytics: {T['eval_header']}")
    
    # Intro text - always visible
    st.markdown(T["eval_info_content"])
    st.markdown(T["terms_explain"])
    
    highlight = results.get("rl_ppo", next(iter(results.values())))
    rl_key = "rl_ppo" if "rl_ppo" in results else list(results.keys())[0]
    
    # KPI Cards with target hints
    cols = st.columns(4)
    cols[0].metric(
        T["avg_latency"],
        f"{highlight['avg_latency']:.1f} ms",
    )
    cols[1].metric(
        T["p95_latency"],
        f"{highlight['p95_latency']:.1f} ms",
        delta=T["target_label"].format(val="120 ms"),
        delta_color="off",
    )
    cols[2].metric(
        T["energy_mbps"],
        f"{highlight['avg_energy_per_mbps']:.4f}",
    )
    cols[3].metric(
        T["sla_viol"],
        f"{highlight['sla_violation_rate']*100:.1f}%",
        delta=T["target_label"].format(val="<5%"),
        delta_color="off",
    )
    
    # ---- Comparison table ----
    st.subheader(T["comparison"])
    
    # Find the winner
    winner_key = max(results.keys(), key=lambda k: results[k]["avg_reward"])
    winner_label = _plabel(winner_key, lang)
    
    rows = []
    for name in POLICY_KEYS:
        if name not in results:
            continue
        data = results[name]
        policy_display = _plabel(name, lang)
        if name == winner_key:
            policy_display = f"{policy_display} 🏆"
        rows.append({
            T["col_policy"]: policy_display,
            T["col_avg_reward"]: f"{data['avg_reward']:.2f}",
            T["col_avg_lat"]: f"{data['avg_latency']:.1f}",
            T["col_p95_lat"]: f"{data['p95_latency']:.1f}",
            T["col_energy"]: f"{data['avg_energy_per_mbps']:.4f}",
            T["col_sla"]: f"{data['sla_violation_rate']*100:.1f}%",
        })
    
    df_styled = pd.DataFrame(rows)
    
    def highlight_winner(row):
        if "🏆" in str(row[T["col_policy"]]):
            return ["background-color: #d4edda; color: #155724; font-weight: 600"] * len(row)
        return [""] * len(row)
    
    st.dataframe(df_styled.style.apply(highlight_winner, axis=1), use_container_width=True, hide_index=True)
    
    # ---- Trade-off scatter ----
    st.subheader(T["tradeoff_title"])
    fig_trade = go.Figure()
    for i, name in enumerate(POLICY_KEYS):
        if name not in results:
            continue
        data = results[name]
        c = POLICY_COLORS[i]
        label = _plabel(name, lang)
        fig_trade.add_trace(go.Scatter(
            x=[data["avg_energy_per_mbps"]],
            y=[data["avg_latency"]],
            mode="markers+text",
            marker=dict(size=24, color=c, line=dict(width=2, color="#fff")),
            text=[label],
            textposition="top center",
            textfont=dict(size=13, color=c, family="Segoe UI, Arial"),
            name=label,
        ))
    fig_trade.update_layout(
        xaxis=dict(title=dict(text=T["tradeoff_x"], font=dict(size=AXIS_FONT_SIZE)),
                   tickfont=dict(size=TICK_FONT_SIZE)),
        yaxis=dict(title=dict(text=T["tradeoff_y"], font=dict(size=AXIS_FONT_SIZE)),
                   tickfont=dict(size=TICK_FONT_SIZE)),
        font=CHART_FONT,
        legend=dict(font=dict(size=LEGEND_FONT_SIZE)),
        height=400,
        margin=dict(t=20, b=50, l=50, r=20),
        plot_bgcolor="#fafafa",
    )
    st.plotly_chart(fig_trade, use_container_width=True)
    st.caption(T["tradeoff_explain"])
else:
    st.warning(T["no_results"])

# ---------------------------------------------------------------------------
# 2) Live simulation
# ---------------------------------------------------------------------------
st.header(f":material/play_circle: {T['live_header']}")
st.caption(T["live_intro"])

col_left, col_right = st.columns([1, 3])

_policy_display = {_plabel(k, lang): k for k in POLICY_KEYS}

with col_left:
    live_display = st.selectbox(T["policy_label"], list(_policy_display.keys()))
    live_policy = _policy_display[live_display]
    live_seed = st.number_input(T["seed_label"], value=99, min_value=0, max_value=9999)
    run_btn = st.button(f":material/play_arrow: {T['run_btn']}", type="primary")

if run_btn:
    st.session_state["live_steps"] = run_live_episode(live_policy, seed=int(live_seed))
    st.session_state["live_policy_display"] = live_display

if "live_steps" in st.session_state:
    steps = st.session_state["live_steps"]
    live_display_saved = st.session_state.get("live_policy_display", live_display)

    with col_right:
        t_vals = list(range(1, len(steps) + 1))
        latencies = [s["latency_ms"] for s in steps]
        energies = [s["energy_per_mbps"] for s in steps]

        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(
            x=t_vals, y=latencies, mode="lines+markers",
            name=T["live_y_lat"],
            line=dict(color="#0d6efd", width=2),
            marker=dict(size=5),
        ))
        fig_live.add_trace(go.Scatter(
            x=t_vals, y=energies, mode="lines+markers",
            name=T["live_y_eng"],
            yaxis="y2",
            line=dict(color="#fd7e14", width=2),
            marker=dict(size=5),
        ))
        fig_live.update_layout(
            title=dict(text=T["live_chart_title"].format(policy=live_display_saved), font=dict(size=14)),
            xaxis=dict(title=dict(text=T["live_x"], font=dict(size=AXIS_FONT_SIZE)),
                       tickfont=dict(size=TICK_FONT_SIZE)),
            yaxis=dict(title=dict(text=T["live_y_lat"], font=dict(size=AXIS_FONT_SIZE, color="#0d6efd")),
                       tickfont=dict(size=TICK_FONT_SIZE), side="left"),
            yaxis2=dict(title=dict(text=T["live_y_eng"], font=dict(size=AXIS_FONT_SIZE, color="#fd7e14")),
                        tickfont=dict(size=TICK_FONT_SIZE), overlaying="y", side="right"),
            font=CHART_FONT,
            legend=dict(font=dict(size=12), x=0.01, y=0.99),
            height=350,
            margin=dict(t=40, b=50),
            plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_live, use_container_width=True)

    # ---- Last 10 decisions ----
    st.subheader(T["last_n"])
    st.caption(T["last_n_explain"])
    table_data = []
    for s in steps[-10:]:
        table_data.append({
            T["col_step"]: s["t"],
            T["col_target"]: _tlabel(s["target"], lang),
            T["col_lat"]: f"{s['latency_ms']:.1f}",
            T["col_eng"]: f"{s['energy_per_mbps']:.4f}",
            T["col_sla_step"]: T["sla_fail"] if s["sla_violation"] else T["sla_ok"],
            T["col_reward"]: f"{s['reward']:.3f}",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    total_reward = sum(s["reward"] for s in steps)
    avg_lat = np.mean([s["latency_ms"] for s in steps])
    sla_rate = np.mean([s["sla_violation"] for s in steps])

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #28a745, #218838); color: white; padding: 1rem 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
        <h3 style="margin: 0 0 0.3rem 0; font-size: 16px; font-weight: 600;">{T['ep_done_title']}</h3>
        <p style="margin: 0; font-size: 14px;">{T['ep_done_detail'].format(reward=total_reward, lat=avg_lat, sla=sla_rate * 100)}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3) RL vs Baseline comparison
# ---------------------------------------------------------------------------
st.header(f":material/compare_arrows: {T['cmp_header']}")
st.caption(T["cmp_intro"])

cmp_seed = st.number_input(T["cmp_seed"], value=42, min_value=0, max_value=9999, key="cmp_seed")
cmp_btn = st.button(f":material/play_arrow: {T['cmp_btn']}", key="cmp_btn")

if cmp_btn:
    fig_cmp = go.Figure()
    for i, pkey in enumerate(POLICY_KEYS):
        steps = run_live_episode(pkey, seed=int(cmp_seed))
        cum_reward = np.cumsum([s["reward"] for s in steps])
        fig_cmp.add_trace(go.Scatter(
            x=list(range(1, len(cum_reward) + 1)),
            y=cum_reward.tolist(),
            mode="lines",
            name=_plabel(pkey, lang),
            line=dict(color=POLICY_COLORS[i], width=2.5),
        ))

    fig_cmp.update_layout(
        title=dict(text=T["cmp_title"], font=dict(size=14)),
        xaxis=dict(title=dict(text=T["cmp_x"], font=dict(size=AXIS_FONT_SIZE)),
                   tickfont=dict(size=TICK_FONT_SIZE)),
        yaxis=dict(title=dict(text=T["cmp_y"], font=dict(size=AXIS_FONT_SIZE)),
                   tickfont=dict(size=TICK_FONT_SIZE)),
        font=CHART_FONT,
        legend=dict(font=dict(size=12)),
        height=400,
        margin=dict(t=40, b=50),
        plot_bgcolor="#fafafa",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)
