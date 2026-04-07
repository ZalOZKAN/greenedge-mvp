"""GreenEdge-5G  Streamlit Dashboard.

Run:
    streamlit run greenedge/dashboard/app.py
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path

# --- Ensure repo root is on sys.path so 'greenedge' package is importable ---
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from greenedge.rl.baselines import greedy_min_energy, greedy_min_latency, simple_threshold
from greenedge.simulator.config import EnvConfig
from greenedge.simulator.env import GreenEdgeEnv

# ---------------------------------------------------------------------------
# POLICY_MAP — Tek kaynak, her yerde bu kullanılacak
# ---------------------------------------------------------------------------
POLICY_MAP = {
    "rl_ppo": {"tr": "PPO", "en": "PPO"},
    "greedy_min_latency": {"tr": "Hız", "en": "Speed"},
    "greedy_min_energy": {"tr": "Maliyet", "en": "Cost"},
    "simple_threshold": {"tr": "Yük", "en": "Load"},
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
        "eval_info_content": "PPO (Proximal Policy Optimization) modelimizin 200 test senaryosu üzerindeki performansı. Amaç: en düşük gecikme ve en az enerji tüketimi ile iş yüklerini yönlendirmek, SLA ihlallerini sıfıra yakın tutmak.",
        "terms_explain": "**Terimler:** **PPO** = Proximal Policy Optimization (Pekiştirmeli Öğrenme modeli) · **Yük** = CPU yoğunluğuna dayalı basit kural · **P95** = 100 istekten 95'inin bu sürede tamamlandığı gecikme · **SLA** = Hizmet Seviyesi Anlaşması (maks. 120 ms).",
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
        "ep_done_title": "SİMÜLASYON (50 ADIM) TAMAMLANDI",
        "ep_done_note": "Bu sonuçlar son çalıştırılan tek simülasyon senaryosuna aittir.",
        "sim_running": "Simülasyon çalışıyor...",
        "cmp_running": "Karşılaştırma yapılıyor...",
        "eval_sub": "Genel Değerlendirme (200 episode)",
        "eval_table_note": "Aşağıdaki kümülatif tablo, modele ait 200 rastgele test senaryosu (episode) üzerinden hesaplanan genel ortalama sonuçları göstermektedir.",
        "tradeoff_note": "Bu grafik de yine 200 farklı test senaryosunun (episode) ortalamalarını temel alarak oluşmuştur.",
        "lbl_policy": "Politika",
        "lbl_total_steps": "Toplam Adım",
        "lbl_tot_reward": "Toplam Ödül",
        "lbl_avg_lat": "Ortalama Gecikme",
        "lbl_avg_eng": "Ortalama Enerji",
        "lbl_sla_rate": "SLA İhlal Oranı",
        "cmp_header": "Politika Karşılaştırması",
        "cmp_intro": "Aynı senaryoda tüm politikaları karşılaştırın. Üstte kalan eğri (sıfıra yakın) = daha iyi performans.",
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
        "story_panel": "Bu ekran, <strong>gecikme (latency)</strong> ile <strong>enerji tüketimi</strong> arasındaki dengeyi gösterir. 5G kenar sunucuları düşük gecikme sağlarken yüksek enerji harcayabilir; bulut ucuzdur ama yavaştır. <strong>PPO politikası</strong>, pekiştirmeli öğrenme ile bu iki hedefi aynı anda optimize eder — SLA ihlallerini sıfıra yakın tutarken enerji verimliliğini korur.",
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
        "eval_info_content": "PPO (Proximal Policy Optimization) model performance across 200 test scenarios. Goal: route workloads with lowest latency and minimal energy, keeping SLA violations near zero.",
        "terms_explain": "**Terms:** **PPO** = Proximal Policy Optimization (Reinforcement Learning agent) · **Load** = simple CPU-threshold heuristic · **P95** = 95th percentile latency (95% of requests complete within this time) · **SLA** = Service Level Agreement (max 120 ms).",
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
        "ep_done_title": "SIMULATION (50 STEPS) COMPLETE",
        "ep_done_note": "These results belong to the single simulation scenario just executed.",
        "sim_running": "Simulation running...",
        "cmp_running": "Running comparison...",
        "eval_sub": "Overall Evaluation (200 episodes)",
        "eval_table_note": "The table below shows the overall average results obtained across 200 random test scenarios (episodes).",
        "tradeoff_note": "This chart is also based on the averages across 200 different test scenarios.",
        "lbl_policy": "Policy",
        "lbl_total_steps": "Total Steps",
        "lbl_tot_reward": "Total Reward",
        "lbl_avg_lat": "Avg Latency",
        "lbl_avg_eng": "Avg Energy",
        "lbl_sla_rate": "SLA Violation Rate",
        "cmp_header": "Policy Comparison",
        "cmp_intro": "Compare all policies on same scenario. Higher curve (closer to zero) = better performance.",
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
        "story_panel": "This screen shows the trade-off between <strong>latency</strong> and <strong>energy consumption</strong>. 5G edge servers offer low latency but consume more energy; cloud is cheaper but slower. The <strong>PPO policy</strong> uses reinforcement learning to optimize both goals simultaneously — keeping SLA violations near zero while preserving energy efficiency.",
    },
}

TARGET_LABELS = {
    "tr": {"edge-a": "Kenar-A", "edge-b": "Kenar-B", "cloud": "Bulut"},
    "en": {"edge-a": "Edge-A", "edge-b": "Edge-B", "cloud": "Cloud"},
}

# ---------------------------------------------------------------------------
# Typography & Chart Constants
# ---------------------------------------------------------------------------
CHART_FONT = {"family": "Segoe UI, Arial, sans-serif", "size": 14, "color": "#212529"}
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


def load_results() -> dict:
    """Load saved results as fallback."""
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _quick_evaluate(policy_fn, n_episodes: int = 30, seed: int | None = None) -> dict:
    """Run a quick evaluation (fewer episodes) and return KPI dict."""
    if seed is None:
        seed = random.randint(0, 999_999)
    cfg = EnvConfig(seed=seed)
    env = GreenEdgeEnv(config=cfg)

    episode_rewards = []
    all_latencies = []
    all_energies = []
    all_sla = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            all_latencies.append(info["latency_ms"])
            all_energies.append(info["energy_per_mbps"])
            all_sla.append(info["sla_violation"])
            done = terminated or truncated
        episode_rewards.append(ep_reward)

    lat = np.array(all_latencies)
    eng = np.array(all_energies)
    sla = np.array(all_sla)

    return {
        "avg_reward": round(float(np.mean(episode_rewards)), 4),
        "std_reward": round(float(np.std(episode_rewards)), 4),
        "avg_latency": round(float(np.mean(lat)), 2),
        "p95_latency": round(float(np.percentile(lat, 95)), 2),
        "avg_energy_per_mbps": round(float(np.mean(eng)), 4),
        "sla_violation_rate": round(float(np.mean(sla)), 4),
        "episode_rewards": [round(r, 4) for r in episode_rewards],
    }


@st.cache_data
def generate_live_results(seed: int | None = None) -> dict:
    """Run all 4 policies on fresh random scenarios and return results dict."""
    if seed is None:
        seed = 42 # Use a fixed seed by default for consistent global results layout

    policy_fns = {
        "rl_ppo": _rl_predict,
        "greedy_min_latency": greedy_min_latency,
        "greedy_min_energy": greedy_min_energy,
        "simple_threshold": simple_threshold,
    }

    results = {}
    for key, fn in policy_fns.items():
        results[key] = _quick_evaluate(fn, n_episodes=200, seed=seed)
    return results


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


def run_live_episode(policy_name: str, seed: int | None = None) -> list[dict]:
    if seed is None:
        seed = random.randint(0, 999_999)
    cfg = EnvConfig(seed=seed)
    env = GreenEdgeEnv(config=cfg)
    obs, _ = env.reset()
    policy_fn = {
        "rl_ppo": _rl_predict,
        "greedy_min_latency": greedy_min_latency,
        "greedy_min_energy": greedy_min_energy,
        "simple_threshold": simple_threshold,
    }[policy_name]
    steps: list[dict] = []
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
def _git_commit_hash() -> str:
    """Return short git commit hash or 'N/A'."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return "N/A"


def generate_pdf(results: dict, lang: str) -> bytes:
    """Generate comprehensive PDF report with Turkish character support and inline charts."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        HRFlowable,
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    T = TEXTS[lang]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.8 * cm, bottomMargin=1.5 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )

    # --- Register a Unicode-capable font for Turkish characters ---
    _font_registered = False
    for font_path_candidate in [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]:
        try:
            pdfmetrics.registerFont(TTFont("UniFont", font_path_candidate))
            _font_registered = True
            break
        except Exception:
            continue

    # Bold variant
    for bold_candidate in [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/tahomabd.ttf",
    ]:
        try:
            pdfmetrics.registerFont(TTFont("UniFont-Bold", bold_candidate))
            break
        except Exception:
            continue

    FONT = "UniFont" if _font_registered else "Helvetica"
    FONT_BOLD = "UniFont-Bold" if _font_registered else "Helvetica-Bold"

    styles = getSampleStyleSheet()

    # Custom styles with Unicode font
    title_style = ParagraphStyle(
        'PDFTitle', parent=styles['Heading1'],
        fontName=FONT_BOLD, fontSize=26, spaceAfter=6,
        textColor=colors.HexColor("#0d6efd"),
    )
    subtitle_style = ParagraphStyle(
        'PDFSubtitle', parent=styles['Normal'],
        fontName=FONT, fontSize=12, spaceAfter=12,
        textColor=colors.HexColor("#6c757d"),
    )
    h2_style = ParagraphStyle(
        'PDFH2', parent=styles['Heading2'],
        fontName=FONT_BOLD, fontSize=16, spaceBefore=18, spaceAfter=8,
        textColor=colors.HexColor("#212529"),
        borderWidth=0, borderPadding=0,
    )
    ParagraphStyle(
        'PDFH3', parent=styles['Heading3'],
        fontName=FONT_BOLD, fontSize=13, spaceBefore=12, spaceAfter=6,
        textColor=colors.HexColor("#495057"),
    )
    body_style = ParagraphStyle(
        'PDFBody', parent=styles['Normal'],
        fontName=FONT, fontSize=11, spaceAfter=6, leading=15,
    )
    small_style = ParagraphStyle(
        'PDFSmall', parent=styles['Normal'],
        fontName=FONT, fontSize=9, textColor=colors.HexColor("#6c757d"), spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        'PDFBullet', parent=styles['Normal'],
        fontName=FONT, fontSize=11, spaceAfter=3, leading=14,
        leftIndent=15, bulletIndent=5,
    )

    story = []

    # =====================================================================
    # PAGE 1: Title + Project Overview + Config
    # =====================================================================
    story.append(Paragraph("GreenEdge-5G", title_style))
    if lang == "tr":
        story.append(Paragraph(
            "5G kenar/bulut altyapisi icin yapay zeka destekli is yuku yonlendirme sistemi",
            subtitle_style,
        ))
    else:
        story.append(Paragraph(
            "AI-powered workload routing for 5G edge/cloud infrastructure",
            subtitle_style,
        ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0d6efd")))
    story.append(Spacer(1, 8))

    # Meta info
    meta_data = [
        [
            Paragraph(f"<b>{T['pdf_generated']}:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style),
            Paragraph(f"<b>Git:</b> {_git_commit_hash()}", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[9 * cm, 7 * cm])
    meta_table.setStyle(TableStyle([
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # --- Project description ---
    if lang == "tr":
        desc_title = "Proje Hakkinda"
        desc_text = (
            "GreenEdge-5G, 5G sebekelerinde is yuku yonlendirme kararlarini "
            "yapay zeka ile optimize eden bir karar motorudur. Sistem, enerji tuketimi "
            "ve gecikme (latency) arasindaki dengeyi Pekistirmeli Ogrenme (Reinforcement Learning) "
            "ile yonetir. Bu rapor, farkli politikalarin simulasyon ortamindaki performansini karsilastirir."
        )
    else:
        desc_title = "About the Project"
        desc_text = (
            "GreenEdge-5G is a decision engine that optimizes workload routing in 5G networks "
            "using AI. The system manages the trade-off between energy consumption and latency "
            "through Reinforcement Learning. This report compares performance of different policies "
            "in a simulated environment."
        )
    story.append(Paragraph(desc_title, h2_style))
    story.append(Paragraph(desc_text, body_style))
    story.append(Spacer(1, 10))

    # --- Environment Config ---
    cfg = EnvConfig()
    config_lbl = "Ortam Yapilandirmasi" if lang == "tr" else "Environment Config"
    story.append(Paragraph(config_lbl, h2_style))
    cfg_data = [
        [Paragraph("<b>Parametre</b>", body_style), Paragraph("<b>Deger</b>", body_style)],
        ["Episode uzunlugu" if lang == "tr" else "Episode length", str(cfg.episode_length)],
        ["SLA esigi" if lang == "tr" else "SLA threshold", f"{cfg.sla_ms} ms"],
        ["Odul alpha (enerji)" if lang == "tr" else "Reward alpha (energy)", str(cfg.reward.alpha)],
        ["Odul beta (gecikme)" if lang == "tr" else "Reward beta (latency)", str(cfg.reward.beta)],
        ["Odul gamma (SLA)" if lang == "tr" else "Reward gamma (SLA)", str(cfg.reward.gamma)],
    ]
    cfg_table = Table(cfg_data, colWidths=[9 * cm, 7 * cm])
    cfg_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), FONT),
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
    ]))
    story.append(cfg_table)
    story.append(Spacer(1, 10))

    # --- Reward formula ---
    formula_title = "Odul Fonksiyonu" if lang == "tr" else "Reward Function"
    story.append(Paragraph(formula_title, h2_style))
    story.append(Paragraph(
        "Reward = -( alpha x Energy_norm + beta x Latency_norm + gamma x SLA_penalty )",
        ParagraphStyle('Formula', parent=body_style, fontName='Courier', fontSize=11,
                       backColor=colors.HexColor("#f1f3f5"), borderPadding=6),
    ))
    if lang == "tr":
        story.append(Paragraph(
            "Sistem, enerji verimliligi ile gecikme performansini birlikte optimize eder. "
            "SLA esigini asan kararlar ek ceza alir.",
            body_style,
        ))
    else:
        story.append(Paragraph(
            "The system jointly optimizes energy efficiency and latency performance. "
            "Decisions exceeding the SLA threshold receive additional penalty.",
            body_style,
        ))

    # --- Policy descriptions ---
    pol_title = "Politikalar" if lang == "tr" else "Policies"
    story.append(Paragraph(pol_title, h2_style))
    if lang == "tr":
        pol_descriptions = [
            "<b>PPO:</b> Proximal Policy Optimization - Pekistirmeli ogrenme ile dinamik optimizasyon",
            "<b>Hiz:</b> Her zaman en dusuk gecikmeli sunucuyu secer (greedy)",
            "<b>Maliyet:</b> Her zaman en dusuk enerjili sunucuyu secer (greedy)",
            "<b>Yuk:</b> CPU yukune gore esik tabanli yonlendirme (threshold-based)",
        ]
    else:
        pol_descriptions = [
            "<b>PPO:</b> Proximal Policy Optimization - Dynamic optimization via RL",
            "<b>Speed:</b> Always picks lowest latency target (greedy)",
            "<b>Cost:</b> Always picks lowest energy target (greedy)",
            "<b>Load:</b> CPU threshold-based routing heuristic",
        ]
    for pd_text in pol_descriptions:
        story.append(Paragraph(pd_text, bullet_style, bulletText="\u2022"))

    # =====================================================================
    # PAGE 2: KPI Results + Comparison Table
    # =====================================================================
    story.append(PageBreak())
    story.append(Paragraph(T["pdf_kpi_section"], title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0d6efd")))
    story.append(Spacer(1, 12))

    ppo_data = results.get("rl_ppo", {})
    if ppo_data:
        story.append(Paragraph(
            "PPO Model KPI" if lang == "en" else "PPO Model Performansi",
            h2_style,
        ))

        l_lat = "Ort. Gecikme" if lang == "tr" else "Avg Latency"
        l_p95 = "P95 Gecikme" if lang == "tr" else "P95 Latency"
        l_sla = "SLA Ihlal %" if lang == "tr" else "SLA Violation %"
        l_eng = "Enerji/Mbps" if lang == "tr" else "Energy/Mbps"
        l_rew = "Ort. Odul" if lang == "tr" else "Avg Reward"

        kpi_data = [
            [
                Paragraph(f"<b><font color='#6c757d'>{l_lat}</font></b><br/><br/><font size=16 color='#212529'><b>{ppo_data.get('avg_latency', 0):.1f} ms</b></font>", body_style),
                Paragraph(f"<b><font color='#6c757d'>{l_p95}</font></b><br/><br/><font size=16 color='#212529'><b>{ppo_data.get('p95_latency', 0):.1f} ms</b></font>", body_style),
                Paragraph(f"<b><font color='#6c757d'>{l_sla}</font></b><br/><br/><font size=16 color='#212529'><b>{ppo_data.get('sla_violation_rate', 0) * 100:.1f}%</b></font>", body_style)
            ],
            [
                Paragraph(f"<b><font color='#6c757d'>{l_eng}</font></b><br/><br/><font size=16 color='#212529'><b>{ppo_data.get('avg_energy_per_mbps', 0):.4f}</b></font>", body_style),
                Paragraph(f"<b><font color='#6c757d'>{l_rew}</font></b><br/><br/><font size=16 color='#212529'><b>{ppo_data.get('avg_reward', 0):.2f}</b></font>", body_style),
                Paragraph(f"<b><font color='#6c757d'>Model</font></b><br/><br/><font size=16 color='#212529'><b>PPO</b></font>", body_style)
            ]
        ]
        kpi_table = Table(kpi_data, colWidths=[5 * cm, 5 * cm, 5 * cm])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
            ('GRID', (0, 0), (-1, -1), 2, colors.white),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#e9ecef")),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 16))

    # --- Full comparison table ---
    story.append(Paragraph(T["pdf_comparison_section"], h2_style))

    col_names = [
        T["col_policy"], T["col_avg_reward"],
        "Ort. Gecikme" if lang == "tr" else "Avg Lat",
        "P95" if lang == "tr" else "P95",
        "Enerji" if lang == "tr" else "Energy",
        "SLA %" if lang == "tr" else "SLA %",
    ]
    header_row = [Paragraph(f"<b>{c}</b>", body_style) for c in col_names]
    table_data = [header_row]

    winner_key = max(results.keys(), key=lambda k: results[k]["avg_reward"]) if results else None

    for key in POLICY_KEYS:
        if key in results:
            data = results[key]
            label = _plabel(key, lang)
            if key == winner_key:
                label = f"{label}  (Kazanan)" if lang == "tr" else f"{label}  (Winner)"
            row = [
                label,
                f"{data['avg_reward']:.2f}",
                f"{data['avg_latency']:.1f}",
                f"{data['p95_latency']:.1f}",
                f"{data['avg_energy_per_mbps']:.4f}",
                f"{data['sla_violation_rate'] * 100:.1f}%",
            ]
            table_data.append(row)

    comp_table = Table(table_data, colWidths=[4 * cm, 2.5 * cm, 2.5 * cm, 2 * cm, 2.5 * cm, 2.5 * cm])

    tbl_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), FONT),
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
    ]

    if winner_key and winner_key in POLICY_KEYS:
        winner_idx = POLICY_KEYS.index(winner_key) + 1
        tbl_style.append(('BACKGROUND', (0, winner_idx), (-1, winner_idx), colors.HexColor("#d4edda")))

    comp_table.setStyle(TableStyle(tbl_style))
    story.append(comp_table)
    story.append(Spacer(1, 16))

    # --- Winner announcement ---
    if winner_key:
        winner_label = _plabel(winner_key, lang)
        if lang == "tr":
            winner_text = f"En iyi politika: <b>{winner_label}</b> (en yuksek ortalama odul)"
        else:
            winner_text = f"Best policy: <b>{winner_label}</b> (highest average reward)"
        story.append(Paragraph(
            winner_text,
            ParagraphStyle('WinnerText', parent=body_style, fontSize=14,
                           textColor=colors.HexColor("#155724"),
                           backColor=colors.HexColor("#d4edda"),
                           borderPadding=8),
        ))

    # =====================================================================
    # PAGE 3: Inline Charts (generated on the fly)
    # =====================================================================
    story.append(PageBreak())
    chart_title = "Grafikler" if lang == "tr" else "Charts"
    story.append(Paragraph(chart_title, title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0d6efd")))
    story.append(Spacer(1, 12))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Chart 1: Episode Rewards bar chart
    story.append(Paragraph(
        "Ortalama Odul Karsilastirmasi" if lang == "tr" else "Average Reward Comparison",
        h2_style,
    ))

    fig1, ax1 = plt.subplots(figsize=(7, 3.5))
    pol_labels = []
    pol_rewards = []
    pol_colors_mpl = []
    for i, key in enumerate(POLICY_KEYS):
        if key in results:
            pol_labels.append(_plabel(key, lang))
            pol_rewards.append(results[key]["avg_reward"])
            pol_colors_mpl.append(POLICY_COLORS[i])

    bars = ax1.barh(pol_labels, pol_rewards, color=pol_colors_mpl, edgecolor="white", height=0.5)
    ax1.set_xlabel("Ortalama Odul" if lang == "tr" else "Average Reward", fontsize=11)
    ax1.set_title("Politika Karsilastirmasi" if lang == "tr" else "Policy Comparison", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, pol_rewards):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=10)
    ax1.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    buf1.seek(0)
    story.append(Image(buf1, width=15 * cm, height=7.5 * cm))
    story.append(Spacer(1, 16))

    # Chart 2: Latency vs Energy scatter
    story.append(Paragraph(
        "Gecikme - Enerji Dengesi" if lang == "tr" else "Latency - Energy Trade-off",
        h2_style,
    ))

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for i, key in enumerate(POLICY_KEYS):
        if key in results:
            data = results[key]
            ax2.scatter(
                data["avg_energy_per_mbps"], data["avg_latency"],
                s=200, c=POLICY_COLORS[i], edgecolors="white", linewidth=2, zorder=5,
            )
            ax2.annotate(
                _plabel(key, lang),
                (data["avg_energy_per_mbps"], data["avg_latency"]),
                textcoords="offset points", xytext=(10, 8), fontsize=11,
                fontweight="bold", color=POLICY_COLORS[i],
            )
    ax2.set_xlabel("Enerji / Mbps" if lang == "tr" else "Energy / Mbps", fontsize=11)
    ax2.set_ylabel("Gecikme (ms)" if lang == "tr" else "Latency (ms)", fontsize=11)
    ax2.set_title(
        "Ideal konum: sol-alt kose" if lang == "tr" else "Ideal: bottom-left corner",
        fontsize=11, style="italic", color="#6c757d",
    )
    ax2.grid(alpha=0.3)
    plt.tight_layout()

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    buf2.seek(0)
    story.append(Image(buf2, width=15 * cm, height=8.5 * cm))
    story.append(Spacer(1, 16))

    # Chart 3: Latency & SLA comparison grouped bar
    story.append(Paragraph(
        "Gecikme ve SLA Karsilastirmasi" if lang == "tr" else "Latency & SLA Comparison",
        h2_style,
    ))

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(7, 3.5))
    names = []
    avg_lats = []
    p95_lats = []
    sla_rates = []
    bar_colors = []
    for i, key in enumerate(POLICY_KEYS):
        if key in results:
            names.append(_plabel(key, lang))
            avg_lats.append(results[key]["avg_latency"])
            p95_lats.append(results[key]["p95_latency"])
            sla_rates.append(results[key]["sla_violation_rate"] * 100)
            bar_colors.append(POLICY_COLORS[i])

    x_pos = range(len(names))
    ax3a.bar(x_pos, avg_lats, color=bar_colors, alpha=0.7, label="Ort." if lang == "tr" else "Avg")
    ax3a.bar(x_pos, p95_lats, color=bar_colors, alpha=0.3, label="P95")
    ax3a.set_xticks(x_pos)
    ax3a.set_xticklabels(names, fontsize=9)
    ax3a.set_ylabel("ms", fontsize=10)
    ax3a.set_title("Gecikme" if lang == "tr" else "Latency", fontsize=11, fontweight="bold")
    ax3a.axhline(y=cfg.sla_ms, color="red", linestyle="--", alpha=0.5, label=f"SLA ({cfg.sla_ms}ms)")
    ax3a.legend(fontsize=8)
    ax3a.grid(axis="y", alpha=0.3)

    ax3b.bar(x_pos, sla_rates, color=bar_colors)
    ax3b.set_xticks(x_pos)
    ax3b.set_xticklabels(names, fontsize=9)
    ax3b.set_ylabel("%", fontsize=10)
    ax3b.set_title("SLA Ihlal" if lang == "tr" else "SLA Violation", fontsize=11, fontweight="bold")
    ax3b.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="Hedef < 5%")
    ax3b.legend(fontsize=8)
    ax3b.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    buf3 = io.BytesIO()
    fig3.savefig(buf3, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    buf3.seek(0)
    story.append(Image(buf3, width=16 * cm, height=7 * cm))

    # =====================================================================
    # PAGE 4: Methodology + Conclusion
    # =====================================================================
    story.append(PageBreak())
    method_title = "Yontem ve Sonuc" if lang == "tr" else "Methodology & Conclusion"
    story.append(Paragraph(method_title, title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0d6efd")))
    story.append(Spacer(1, 12))

    # Architecture overview
    arch_title = "Sistem Mimarisi" if lang == "tr" else "System Architecture"
    story.append(Paragraph(arch_title, h2_style))
    if lang == "tr":
        arch_items = [
            "<b>Simulasyon:</b> Gymnasium tabanli 5G ag ortami (6 boyutlu durum vektoru, 3 eylem)",
            "<b>RL Motoru:</b> Stable-Baselines3 PPO (PyTorch backend)",
            "<b>API:</b> FastAPI ile gercek zamanli karar servisi",
            "<b>Dashboard:</b> Streamlit ile KPI gorsellestirme ve PDF rapor",
            "<b>Deployment:</b> Kubernetes (k3s) uyumlu container mimarisi",
        ]
    else:
        arch_items = [
            "<b>Simulation:</b> Gymnasium-based 5G network env (6-dim state, 3 actions)",
            "<b>RL Engine:</b> Stable-Baselines3 PPO (PyTorch backend)",
            "<b>API:</b> FastAPI real-time decision service",
            "<b>Dashboard:</b> Streamlit with KPI visualization and PDF export",
            "<b>Deployment:</b> Kubernetes (k3s) compatible container architecture",
        ]
    for item in arch_items:
        story.append(Paragraph(item, bullet_style, bulletText="\u2022"))
    story.append(Spacer(1, 12))

    # State/Action description
    sa_title = "Durum ve Eylem Uzayi" if lang == "tr" else "State & Action Space"
    story.append(Paragraph(sa_title, h2_style))
    sa_data = [
        [Paragraph("<b>Indeks</b>", body_style), Paragraph("<b>Degisken</b>", body_style),
         Paragraph("<b>Aciklama</b>", body_style)],
        ["0", "cpu_a", "Edge-A CPU yuku" if lang == "tr" else "Edge-A CPU load"],
        ["1", "cpu_b", "Edge-B CPU yuku" if lang == "tr" else "Edge-B CPU load"],
        ["2", "q_a", "Edge-A kuyruk orani" if lang == "tr" else "Edge-A queue ratio"],
        ["3", "q_b", "Edge-B kuyruk orani" if lang == "tr" else "Edge-B queue ratio"],
        ["4", "link_q", "Baglanti kalitesi" if lang == "tr" else "Link quality"],
        ["5", "energy_price", "Enerji fiyati" if lang == "tr" else "Energy price"],
    ]
    sa_table = Table(sa_data, colWidths=[2 * cm, 3.5 * cm, 10.5 * cm])
    sa_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
    ]))
    story.append(sa_table)
    story.append(Spacer(1, 8))

    action_data = [
        [Paragraph("<b>Eylem</b>", body_style), Paragraph("<b>Hedef</b>", body_style),
         Paragraph("<b>Ozellik</b>", body_style)],
        ["0", "Edge-A", "Dusuk gecikme, yuksek enerji" if lang == "tr" else "Low latency, high energy"],
        ["1", "Edge-B", "Orta gecikme, orta enerji" if lang == "tr" else "Medium latency, medium energy"],
        ["2", "Cloud", "Yuksek gecikme, dusuk enerji" if lang == "tr" else "High latency, low energy"],
    ]
    act_table = Table(action_data, colWidths=[2 * cm, 3.5 * cm, 10.5 * cm])
    act_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#495057")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
    ]))
    story.append(act_table)
    story.append(Spacer(1, 12))

    # Confidence & Fallback
    conf_title = "Guven Skoru ve Fallback" if lang == "tr" else "Confidence Score & Fallback"
    story.append(Paragraph(conf_title, h2_style))
    if lang == "tr":
        story.append(Paragraph(
            "Her karar icin bir guven skoru (0-1) uretilir. "
            "Guven = max_probability - second_max_probability. "
            "Guven skoru esik degerin altina dustugunde sistem, "
            "kural tabanli fallback politikasina gecer (varsayilan: Yuk). "
            "Bu mekanizma endustriyel entegrasyon icin guvenli bir mimari saglar.",
            body_style,
        ))
    else:
        story.append(Paragraph(
            "A confidence score (0-1) is produced for each decision. "
            "Confidence = max_probability - second_max_probability. "
            "When confidence drops below threshold, the system falls back "
            "to a rule-based policy (default: Load). "
            "This mechanism provides a safe architecture for industrial integration.",
            body_style,
        ))
    story.append(Spacer(1, 12))

    # Conclusion
    conclusion_title = "Sonuc" if lang == "tr" else "Conclusion"
    story.append(Paragraph(conclusion_title, h2_style))
    if lang == "tr":
        conclusion_text = (
            "Simulasyon ortaminda yapilan testlerde PPO politikasi, "
            "kural tabanli alternatiflere kiyasla hem enerji tuketiminde hem de "
            "gecikme performansinda tutarli iyilesmeler gostermistir. "
            "Bu sonuclar, yapay zeka tabanli yuk yonlendirmenin 5G sebekelerinde "
            "uygulanabilir oldugunu gostermektedir. Bir sonraki adim: "
            "gercek ortam verileriyle dogrulama."
        )
    else:
        conclusion_text = (
            "In simulation-based testing, the PPO policy showed consistent improvements "
            "in both energy consumption and latency performance compared to rule-based alternatives. "
            "These results demonstrate the feasibility of AI-based workload routing in 5G networks. "
            "Next step: validation with real-world data."
        )
    story.append(Paragraph(conclusion_text, body_style))
    story.append(Spacer(1, 12))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dee2e6")))
    if lang == "tr":
        disclaimer = (
            "<i>Not: Bu rapordaki tum degerler simulasyon ortaminda uretilmistir. "
            "Gercek saha verisi kullanilmamistir. Sonuclar her calistirmada "
            "rastgele senaryolar nedeniyle farklilik gosterebilir.</i>"
        )
    else:
        disclaimer = (
            "<i>Note: All values in this report are generated in a simulated environment. "
            "No real field data was used. Results may vary between runs due to random scenarios.</i>"
        )
    story.append(Paragraph(disclaimer, small_style))
    story.append(Spacer(1, 20))

    # Footer
    story.append(Paragraph(T["pdf_footer"], ParagraphStyle(
        'PDFFooter', parent=styles['Normal'], fontName=FONT,
        fontSize=10, textColor=colors.gray,
    )))

    doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GreenEdge-5G",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "first_page_load" not in st.session_state:
    st.components.v1.html(
        "<script>window.parent.document.querySelector('.main').scrollTo(0,0); window.parent.scrollTo(0,0);</script>",
        height=0,
    )
    st.session_state["first_page_load"] = True

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
# Global CSS — Load from central style module
# ---------------------------------------------------------------------------
_CSS_PATH = Path(__file__).parent / "style.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load results — live evaluation (different every page load)
# ---------------------------------------------------------------------------
# Generate fresh results each time; fall back to saved file if model missing
results = generate_live_results()

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
# Story Panel (P0) — "What this means" explanation
# ---------------------------------------------------------------------------
st.markdown(f'<div class="story-panel">{T["story_panel"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1) KPI Cards — Evaluation Summary
# ---------------------------------------------------------------------------
if results:


    # ---- Comparison table ----
    st.subheader(T["comparison"])
    st.info(T["eval_table_note"])

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
    st.info(T["tradeoff_note"])
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
            marker={"size": 24, "color": c, "line": {"width": 2, "color": "#fff"}},
            text=[label],
            textposition="top center",
            textfont={"size": 13, "color": c, "family": "Segoe UI, Arial"},
            name=label,
        ))
    fig_trade.update_layout(
        xaxis={"title": {"text": T["tradeoff_x"], "font": {"size": AXIS_FONT_SIZE}},
                   "tickfont": {"size": TICK_FONT_SIZE}},
        yaxis={"title": {"text": T["tradeoff_y"], "font": {"size": AXIS_FONT_SIZE}},
                   "tickfont": {"size": TICK_FONT_SIZE}},
        font=CHART_FONT,
        legend={"font": {"size": LEGEND_FONT_SIZE}},
        height=400,
        margin={"t": 20, "b": 50, "l": 50, "r": 20},
        plot_bgcolor="#fafafa",
    )
    st.plotly_chart(fig_trade, use_container_width=True)
    st.caption(T["tradeoff_explain"])
else:
    st.warning(T["no_results"])

# ---------------------------------------------------------------------------
# 3) RL vs Baseline comparison
# ---------------------------------------------------------------------------
st.header(f"⇄ {T['cmp_header']}")
st.caption(T["cmp_intro"])

cmp_col1, cmp_col2 = st.columns([1, 3])
with cmp_col1:
    auto_seed_cmp = st.checkbox("Rastgele senaryo" if lang == "tr" else "Random scenario", value=True, key="auto_seed_cmp")
    if not auto_seed_cmp:
        cmp_seed = st.number_input(T["cmp_seed"], value=42, min_value=0, max_value=9999, key="cmp_seed")
    else:
        cmp_seed = None
    cmp_btn = st.button(f"▶ {T['cmp_btn']}", key="cmp_btn")

if cmp_btn:
    with st.spinner(T["cmp_running"]):
        used_seed = cmp_seed if cmp_seed is not None else random.randint(0, 999_999)
        fig_cmp = go.Figure()
        for i, pkey in enumerate(POLICY_KEYS):
            steps = run_live_episode(pkey, seed=used_seed)
            cum_reward = np.cumsum([s["reward"] for s in steps])
            fig_cmp.add_trace(go.Scatter(
                x=list(range(1, len(cum_reward) + 1)),
                y=cum_reward.tolist(),
                mode="lines",
                name=_plabel(pkey, lang),
                line={"color": POLICY_COLORS[i], "width": 2.5},
            ))

        fig_cmp.update_layout(
            title={"text": T["cmp_title"], "font": {"size": 14}},
            xaxis={"title": {"text": T["cmp_x"], "font": {"size": AXIS_FONT_SIZE}},
                       "tickfont": {"size": TICK_FONT_SIZE}},
            yaxis={"title": {"text": T["cmp_y"], "font": {"size": AXIS_FONT_SIZE}},
                       "tickfont": {"size": TICK_FONT_SIZE}},
            font=CHART_FONT,
            legend={"font": {"size": 12}},
            height=400,
            margin={"t": 40, "b": 50},
            plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

# ---------------------------------------------------------------------------
# 2) Live simulation
# ---------------------------------------------------------------------------
st.header(f"▶ {T['live_header']}")
st.caption(T["live_intro"])

_policy_display = {_plabel(k, lang): k for k in POLICY_KEYS}

# Controls horizontally aligned
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1, 1])
with ctrl_col1:
    live_display = st.selectbox(T["policy_label"], list(_policy_display.keys()))
    live_policy = _policy_display[live_display]
with ctrl_col2:
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    auto_seed = st.checkbox("Rastgele Senaryo Numarası" if lang == "tr" else "Random Scenario Number", value=True, key="auto_seed_live")
    if auto_seed:
        live_seed = None
    else:
        live_seed = st.number_input(T["seed_label"], value=99, min_value=0, max_value=9999, label_visibility="collapsed")
with ctrl_col3:
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    run_btn = st.button(f"▶ {T['run_btn']}", type="primary", use_container_width=True)

if run_btn:
    seed_val = None if auto_seed else int(live_seed)
    with st.spinner(T["sim_running"]):
        st.session_state["live_steps"] = run_live_episode(live_policy, seed=seed_val)
        st.session_state["live_policy_display"] = live_display
        st.session_state["live_seed_used"] = seed_val

if "live_steps" in st.session_state:
    steps = st.session_state["live_steps"]
    live_display_saved = st.session_state.get("live_policy_display", live_display)
    total_reward = sum(s["reward"] for s in steps)
    avg_lat = np.mean([s["latency_ms"] for s in steps])
    avg_eng = np.mean([s["energy_per_mbps"] for s in steps])
    sla_rate = np.mean([s["sla_violation"] for s in steps])

    sla_color = "#198754" if sla_rate == 0 else ("#fd7e14" if sla_rate < 0.05 else "#dc3545")
    
    html_str = f"""
<div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-top: 4px solid #198754; border-radius: 8px; padding: 16px; margin: 16px 0; max-width: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
<h4 style="margin: 0 0 12px 0; font-size: 13.5px; font-weight: 700; color: #495057; text-transform: uppercase;">{T['ep_done_title']}</h4>
<div style="font-size: 13px; color: #212529; display: flex; flex-wrap: wrap; gap: 10px;">
<div style="flex: 1 1 45%; border-bottom: 1px solid #e9ecef; padding-bottom: 4px;"><span style="color: #6c757d; font-size: 12px;">{T['lbl_policy']}</span><br><span style="font-weight: 600;">{live_display_saved}</span></div>
<div style="flex: 1 1 45%; border-bottom: 1px solid #e9ecef; padding-bottom: 4px;"><span style="color: #6c757d; font-size: 12px;">{T['lbl_total_steps']}</span><br><span style="font-weight: 600;">{len(steps)}</span></div>
<div style="flex: 1 1 45%; border-bottom: 1px solid #e9ecef; padding-bottom: 4px;"><span style="color: #6c757d; font-size: 12px;">{T['lbl_tot_reward']}</span><br><span style="font-weight: 600;">{total_reward:.2f}</span></div>
<div style="flex: 1 1 45%; border-bottom: 1px solid #e9ecef; padding-bottom: 4px;"><span style="color: #6c757d; font-size: 12px;">{T['lbl_avg_lat']}</span><br><span style="font-weight: 600; color: #0d6efd;">{avg_lat:.1f} ms</span></div>
<div style="flex: 1 1 45%; padding-bottom: 4px;"><span style="color: #6c757d; font-size: 12px;">{T['lbl_avg_eng']}</span><br><span style="font-weight: 600; color: #fd7e14;">{avg_eng:.4f}</span></div>
<div style="flex: 1 1 45%; padding-bottom: 4px;"><span style="color: #6c757d; font-size: 12px;">{T['lbl_sla_rate']}</span><br><span style="font-weight: 700; color: {sla_color};">{sla_rate * 100:.1f}%</span></div>
</div>
<div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #e9ecef; font-size: 11px; color: #adb5bd; font-style: italic;">{T['ep_done_note']}</div>
</div>
"""

    res_col1, res_col2 = st.columns([0.4, 0.6])
    
    with res_col1:
        st.markdown(html_str, unsafe_allow_html=True)
        
    with res_col2:
        t_vals = list(range(1, len(steps) + 1))
        latencies = [s["latency_ms"] for s in steps]
        energies = [s["energy_per_mbps"] for s in steps]

        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(
            x=t_vals, y=latencies, mode="lines+markers",
            name=T["live_y_lat"],
            line={"color": "#0d6efd", "width": 2},
            marker={"size": 5},
        ))
        fig_live.add_trace(go.Scatter(
            x=t_vals, y=energies, mode="lines+markers",
            name=T["live_y_eng"],
            yaxis="y2",
            line={"color": "#fd7e14", "width": 2},
            marker={"size": 5},
        ))
        fig_live.update_layout(
            title={"text": T["live_chart_title"].format(policy=live_display_saved), "font": {"size": 14}},
            xaxis={"title": {"text": T["live_x"], "font": {"size": AXIS_FONT_SIZE}},
                       "tickfont": {"size": TICK_FONT_SIZE}},
            yaxis={"title": {"text": T["live_y_lat"], "font": {"size": AXIS_FONT_SIZE, "color": "#0d6efd"}},
                       "tickfont": {"size": TICK_FONT_SIZE}, "side": "left"},
            yaxis2={"title": {"text": T["live_y_eng"], "font": {"size": AXIS_FONT_SIZE, "color": "#fd7e14"}},
                        "tickfont": {"size": TICK_FONT_SIZE}, "overlaying": "y", "side": "right"},
            font=CHART_FONT,
            legend={"font": {"size": 12}, "x": 0.01, "y": 0.99},
            height=350,
            margin={"t": 40, "b": 50},
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
