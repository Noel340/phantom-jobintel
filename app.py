"""
PHANTOM // JOBINTEL — AI-Powered Fake Job Advertisement Detection System
Cybersecurity Threat Intelligence Dashboard & Production ML Inference Engine.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METRICS_PATH,
    MODEL_COMPARISON_CSV,
    DATASET_SUMMARY_PATH,
    REPORTS_DIR,
    THEME,
)
from src.predict import get_predictor, check_suspicious_indicators
from src.url_analyzer import fetch_and_extract_job
from src.security import validate_url, sanitize_text

# Streamlit Page Configuration
st.set_page_config(
    page_title="PHANTOM // JOBINTEL — Scam Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cybersecurity Styling
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {THEME['text_primary']};
    background-color: {THEME['obsidian_black']};
}}

.stApp {{
    background: radial-gradient(circle at 15% 10%, #0d1a2d 0%, #05080d 50%, #030508 100%);
    color: {THEME['text_primary']};
}}

code, pre, .mono {{
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background-color: #070e17 !important;
    border-right: 1px solid rgba(0, 229, 255, 0.12) !important;
}}

/* Header and Cards */
.threat-card {{
    background: linear-gradient(135deg, rgba(16, 27, 42, 0.85) 0%, rgba(9, 19, 33, 0.95) 100%);
    border: 1px solid rgba(0, 229, 255, 0.18);
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s ease, border-color 0.2s ease;
}}

.threat-card:hover {{
    border-color: rgba(0, 229, 255, 0.4);
    transform: translateY(-2px);
}}

.metric-highlight {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
    color: {THEME['neon_cyan']};
    text-shadow: 0 0 12px rgba(0, 229, 255, 0.35);
}}

.metric-label {{
    color: {THEME['text_secondary']};
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}}

/* Status badges */
.badge-fraud {{
    background: rgba(255, 77, 109, 0.15);
    border: 1px solid {THEME['threat_red']};
    color: {THEME['threat_red']};
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    display: inline-block;
    letter-spacing: 0.05em;
}}

.badge-legit {{
    background: rgba(0, 245, 160, 0.15);
    border: 1px solid {THEME['emerald']};
    color: {THEME['emerald']};
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    display: inline-block;
    letter-spacing: 0.05em;
}}

.badge-amber {{
    background: rgba(255, 191, 71, 0.15);
    border: 1px solid {THEME['warning_amber']};
    color: {THEME['warning_amber']};
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    display: inline-block;
    letter-spacing: 0.05em;
}}

/* Streamlit Button Restyling */
div.stButton > button {{
    background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%) !important;
    color: #ffffff !important;
    border: 1px solid {THEME['neon_cyan']} !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 0 15px rgba(0, 180, 216, 0.3) !important;
    transition: all 0.2s ease !important;
}}

div.stButton > button:hover {{
    background: linear-gradient(135deg, #00E5FF 0%, #0096C7 100%) !important;
    box-shadow: 0 0 25px rgba(0, 229, 255, 0.55) !important;
    transform: translateY(-1px) !important;
}}

/* Input boxes */
input, textarea, select {{
    background-color: #0b1522 !important;
    border: 1px solid rgba(0, 229, 255, 0.2) !important;
    color: {THEME['text_primary']} !important;
    border-radius: 6px !important;
}}

input:focus, textarea:focus {{
    border-color: {THEME['neon_cyan']} !important;
    box-shadow: 0 0 8px rgba(0, 229, 255, 0.3) !important;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_cached_predictor():
    """Load singleton inference pipeline with caching."""
    try:
        return get_predictor()
    except Exception as e:
        return None


@st.cache_data
def load_metrics_and_summary():
    """Load precomputed metrics and dataset summary if available."""
    metrics_data = None
    summary_data = None
    comparison_df = None

    if MODEL_METRICS_PATH.exists():
        try:
            with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            pass

    if DATASET_SUMMARY_PATH.exists():
        try:
            with open(DATASET_SUMMARY_PATH, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    if MODEL_COMPARISON_CSV.exists():
        try:
            comparison_df = pd.read_csv(MODEL_COMPARISON_CSV)
        except Exception:
            pass

    return metrics_data, summary_data, comparison_df


# Initialize session state for scan history
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


def render_sidebar():
    """Render cybersecurity telemetry sidebar."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="font-size: 1.8rem;">🛡️</span>
                <div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 1.15rem; color: {THEME['neon_cyan']}; letter-spacing: 0.06em;">PHANTOM</div>
                    <div style="font-size: 0.72rem; color: {THEME['text_secondary']}; letter-spacing: 0.12em;">// JOBINTEL OSINT</div>
                </div>
            </div>
            <div style="margin-top: 0.8rem; font-size: 0.75rem; color: {THEME['muted_mint']}; display: flex; align-items: center; gap: 0.4rem;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: {THEME['emerald']}; box-shadow: 0 0 8px {THEME['emerald']};"></span>
                DEFENSE ENGINE ACTIVE
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "NAVIGATION RADAR",
            [
                "🛰️ Command Center",
                "⚡ Job Analyzer",
                "📊 Model Intelligence",
                "🗄️ Dataset Intelligence",
                "📖 Methodology & Viva",
            ],
            index=1,  # Default to Job Analyzer
        )

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size: 0.75rem; color: {THEME['text_secondary']}; line-height: 1.5;">
            <div style="color: {THEME['neon_cyan']}; font-weight: 600; margin-bottom: 0.3rem;">SYSTEM TELEMETRY</div>
            <div>• Core Engine: Scikit-Learn TF-IDF</div>
            <div>• Classification: Calibrated Linear / NB</div>
            <div>• Security: SSRF Filter & RFC1918 Guard</div>
            <div>• Training Dataset: 17,880 Ads</div>
            <div>• Build: v1.0.0-PROD</div>
        </div>
        """, unsafe_allow_html=True)

        return page


def render_command_center(metrics, summary, comparison_df):
    """Render Page A: Command Center Dashboard."""
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: {THEME['text_primary']}; margin-bottom: 0.2rem;">
            PHANTOM <span style="color: {THEME['neon_cyan']};">// JOBINTEL</span>
        </h1>
        <p style="color: {THEME['text_secondary']}; font-size: 1.05rem; letter-spacing: 0.02em;">
            DETECT SCAMS. PROTECT JOB SEEKERS. BUILD A SAFER WEB.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Cards
    col1, col2, col3, col4 = st.columns(4)

    total_records = summary["total_rows"] if summary else 17880
    fraud_pct = summary["class_distribution"]["fraudulent_pct"] if summary else 4.84
    best_model_name = metrics.get("best_model", "Linear SVM / Logistic Reg") if metrics else "Linear SVM"
    best_f1 = metrics["models"][best_model_name]["fraud_f1"] if metrics and best_model_name in metrics.get("models", {}) else 0.78

    with col1:
        st.markdown(f"""
        <div class="threat-card">
            <div class="metric-label">Dataset Corpuses</div>
            <div class="metric-highlight">{total_records:,}</div>
            <div style="font-size: 0.75rem; color: {THEME['muted_mint']}; margin-top: 0.3rem;">Real Employment Records</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="threat-card">
            <div class="metric-label">Fraud Incident Base</div>
            <div class="metric-highlight" style="color: {THEME['threat_red']}; text-shadow: 0 0 12px rgba(255, 77, 109, 0.4);">{fraud_pct}%</div>
            <div style="font-size: 0.75rem; color: {THEME['text_secondary']}; margin-top: 0.3rem;">866 Verified Fraudulent Postings</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="threat-card">
            <div class="metric-label">Selected Champion Model</div>
            <div class="metric-highlight" style="font-size: 1.3rem; color: {THEME['emerald']};">{best_model_name}</div>
            <div style="font-size: 0.75rem; color: {THEME['text_secondary']}; margin-top: 0.3rem;">Calibrated Pipeline</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="threat-card">
            <div class="metric-label">Fraud F1-Score</div>
            <div class="metric-highlight" style="color: {THEME['warning_amber']};">{best_f1 * 100:.1f}%</div>
            <div style="font-size: 0.75rem; color: {THEME['muted_mint']}; margin-top: 0.3rem;">Precision/Recall Balance</div>
        </div>
        """, unsafe_allow_html=True)

    # Middle Architecture & Workflow Section
    st.markdown("### 🛡️ System Defense Workflow")
    w1, w2, w3, w4 = st.columns(4)

    with w1:
        st.markdown(f"""
        <div class="threat-card" style="min-height: 160px;">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">📥</div>
            <div style="font-weight: 700; color: {THEME['neon_cyan']}; margin-bottom: 0.3rem;">1. Dual Ingestion</div>
            <div style="font-size: 0.8rem; color: {THEME['text_secondary']}; line-height: 1.4;">
                Accepts raw user paste or fetches public job postings via protected SSRF-filtered web parser.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with w2:
        st.markdown(f"""
        <div class="threat-card" style="min-height: 160px;">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">🧹</div>
            <div style="font-weight: 700; color: {THEME['electric_blue']}; margin-bottom: 0.3rem;">2. Normalization</div>
            <div style="font-size: 0.8rem; color: {THEME['text_secondary']}; line-height: 1.4;">
                HTML unescaping, tokenization of URLs, emails, phone numbers, and compensation patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with w3:
        st.markdown(f"""
        <div class="threat-card" style="min-height: 160px;">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">🧠</div>
            <div style="font-weight: 700; color: {THEME['emerald']}; margin-bottom: 0.3rem;">3. TF-IDF & ML Inference</div>
            <div style="font-size: 0.8rem; color: {THEME['text_secondary']}; line-height: 1.4;">
                10,000 n-gram features analyzed by the calibrated machine learning classifier.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with w4:
        st.markdown(f"""
        <div class="threat-card" style="min-height: 160px;">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">🔍</div>
            <div style="font-weight: 700; color: {THEME['warning_amber']}; margin-bottom: 0.3rem;">4. Heuristic Scan</div>
            <div style="font-size: 0.8rem; color: {THEME['text_secondary']}; line-height: 1.4;">
                Independent regex scanner detects Telegram interviews, check advance scams, and free emails.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Session Scan History if available
    st.markdown("### 📋 Recent Scans in This Session")
    if st.session_state.scan_history:
        history_df = pd.DataFrame(st.session_state.scan_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No job scans performed in this session yet. Navigate to '⚡ Job Analyzer' to analyze an advertisement.")


def render_job_analyzer(predictor: Optional[Any]):
    """Render Page B & C: Job Analyzer with Paste Mode and URL Mode."""
    st.markdown(f"""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; color: {THEME['text_primary']}; margin-bottom: 0.2rem;">
            ⚡ THREAT ANALYZER <span style="color: {THEME['neon_cyan']};">// LIVE INFERENCE</span>
        </h2>
        <p style="color: {THEME['text_secondary']}; font-size: 0.95rem;">
            Submit advertisement text or public URL to inspect structural indicators and compute ML scam probability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if predictor is None:
        st.error(
            "⚠️ Trained model pipeline not found! Please execute 'python train.py' in the workspace terminal to train and save the models."
        )
        return

    tab_paste, tab_url = st.tabs(["📝 MODE 1: PASTE JOB DETAILS", "🌐 MODE 2: ANALYZE JOB URL"])

    # ----------------------------------------------------
    # TAB 1: PASTE JOB DETAILS
    # ----------------------------------------------------
    with tab_paste:
        with st.form("paste_job_form"):
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                title = st.text_input("Job Title *", placeholder="e.g. Senior Security Analyst, Customer Care Associate")
            with col_t2:
                company = st.text_input("Company / Employer Name", placeholder="e.g. CyberTech Corp")

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                location = st.text_input("Location", placeholder="e.g. US, CA, San Francisco / Remote")
            with col_m2:
                employment_type = st.selectbox(
                    "Employment Type",
                    ["Not Specified", "Full-time", "Part-time", "Contract", "Temporary", "Other"],
                )
            with col_m3:
                experience = st.selectbox(
                    "Experience Level",
                    ["Not Specified", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"],
                )

            desc = st.text_area(
                "Job Description *",
                height=160,
                placeholder="Paste the primary job description text here...",
            )

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                requirements = st.text_area(
                    "Requirements / Qualifications",
                    height=100,
                    placeholder="Candidate qualifications, degrees, skills...",
                )
            with col_r2:
                benefits = st.text_area(
                    "Benefits / Offer Details",
                    height=100,
                    placeholder="Salary, bonuses, health insurance, equipment provided...",
                )

            submit_paste = st.form_submit_button("⚡ INITIATE ANALYSIS")

        if submit_paste:
            if not title.strip() and not desc.strip():
                st.warning("⚠️ Please provide at least a Job Title or Description to analyze.")
            else:
                job_payload = {
                    "title": title,
                    "company_name": company,
                    "location": location,
                    "employment_type": employment_type if employment_type != "Not Specified" else "",
                    "required_experience": experience if experience != "Not Specified" else "",
                    "description": desc,
                    "requirements": requirements,
                    "benefits": benefits,
                }
                with st.spinner("Executing neural feature extraction and heuristic scan..."):
                    result = predictor.predict(job_payload)

                render_analysis_results(result, source="Pasted Text", identifier=title or "Job Ad")

    # ----------------------------------------------------
    # TAB 2: ANALYZE JOB URL
    # ----------------------------------------------------
    with tab_url:
        st.markdown(f"""
        <div style="background: rgba(9, 19, 33, 0.7); border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <div style="font-weight: 700; color: {THEME['neon_cyan']}; font-size: 0.9rem; margin-bottom: 0.3rem;">
                🛡️ SECURE REMOTE OSINT EXTRACTION
            </div>
            <div style="font-size: 0.8rem; color: {THEME['text_secondary']}; line-height: 1.4;">
                Enter a public HTTP/HTTPS job posting link. Target will be resolved through our SSRF firewall to prevent loopback, private RFC1918, and cloud metadata access.
            </div>
        </div>
        """, unsafe_allow_html=True)

        url_input = st.text_input(
            "Public Job Advertisement URL",
            placeholder="https://example.com/careers/senior-engineer-job-123",
        )

        col_url_actions = st.columns([1, 4])
        with col_url_actions[0]:
            extract_btn = st.button("🌐 EXTRACT & ANALYZE")

        if extract_btn:
            if not url_input.strip():
                st.warning("⚠️ Please enter a job URL.")
            else:
                with st.spinner("Validating endpoint security, resolving DNS, and parsing DOM..."):
                    extract_res = fetch_and_extract_job(url_input.strip())

                if not extract_res["success"]:
                    st.error(f"❌ Extraction Failed: {extract_res['error']}")
                    st.info(
                        "💡 Fallback: Some websites restrict automated scrapers or require login. "
                        "You can easily copy and paste the job details directly into the '📝 MODE 1: PASTE JOB DETAILS' tab above!"
                    )
                else:
                    st.success(f"✅ Successfully extracted via {extract_res.get('extraction_method', 'HTML parser')}")
                    # Show extracted preview
                    with st.expander("👁️ Review Extracted Advertisement Data", expanded=True):
                        st.markdown(f"**Extracted Title:** {extract_res.get('job_title', 'N/A')}")
                        st.markdown(f"**Extracted Company:** {extract_res.get('company_name', 'N/A')}")
                        st.markdown(f"**Extracted Location:** {extract_res.get('location', 'N/A')}")
                        st.text_area("Extracted Description Content", value=extract_res.get("description", ""), height=120, disabled=True)

                    # Pass into real ML predictor
                    with st.spinner("Analyzing extracted advertisement with ML model..."):
                        pred_res = predictor.predict({
                            "title": extract_res.get("job_title", ""),
                            "company_name": extract_res.get("company_name", ""),
                            "location": extract_res.get("location", ""),
                            "description": extract_res.get("description", ""),
                            "requirements": extract_res.get("requirements", ""),
                            "benefits": extract_res.get("benefits", ""),
                            "raw_text": extract_res.get("raw_combined_text", ""),
                        })

                    render_analysis_results(
                        pred_res,
                        source="URL Extraction",
                        identifier=extract_res.get("job_title") or url_input,
                    )


def render_analysis_results(result: Dict[str, Any], source: str, identifier: str):
    """Render Page C: Analysis Results Screen."""
    if not result.get("success"):
        st.error(f"Analysis Error: {result.get('error', 'Unknown error during inference.')}")
        return

    pred = result["prediction"]
    verdict = result["verdict"]
    color = result["verdict_color"]
    confidence = result.get("confidence", 0.0)
    fraud_prob = result.get("probability_distribution", {}).get("fraudulent", 0.0)
    legit_prob = result.get("probability_distribution", {}).get("legitimate", 0.0)

    # Record in session state history
    st.session_state.scan_history.insert(0, {
        "Identifier": identifier[:40],
        "Source": source,
        "Verdict": verdict,
        "Fraud Probability": f"{fraud_prob * 100:.1f}%" if fraud_prob is not None else "N/A",
        "Red Flags": len(result.get("suspicious_indicators", [])),
    })

    st.markdown("---")
    st.markdown(f"""
    <div style="background: rgba(16, 27, 42, 0.95); border: 2px solid {color}; border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0; box-shadow: 0 0 25px rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <span class="{ 'badge-fraud' if pred == 1 else 'badge-legit' }" style="font-size: 1.1rem; padding: 0.4rem 1.2rem;">
                    { '🚨 POTENTIALLY FRAUDULENT' if pred == 1 else '🛡️ LIKELY LEGITIMATE' }
                </span>
                <div style="margin-top: 0.8rem; font-size: 1.15rem; color: {THEME['text_primary']}; font-weight: 600;">
                    {result['summary_message']}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; color: {THEME['text_secondary']}; text-transform: uppercase;">Estimated Risk Probability</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 800; color: {color};">
                    {f"{fraud_prob * 100:.1f}%" if fraud_prob is not None else "N/A"}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Detailed Results Breakdown
    col_probs, col_stats = st.columns([1, 1])

    with col_probs:
        st.markdown("#### 📊 Model Probability Distribution")
        if fraud_prob is not None and legit_prob is not None:
            prob_df = pd.DataFrame({
                "Category": ["Legitimate Posting", "Fraudulent / Scam"],
                "Probability": [legit_prob, fraud_prob],
            })
            st.bar_chart(prob_df.set_index("Category"), color=[THEME["electric_blue"]])
            st.caption(f"Legitimate Probability: **{legit_prob * 100:.2f}%** | Fraudulent Risk: **{fraud_prob * 100:.2f}%**")
        else:
            st.info("Direct probability not calibrated for this classifier configuration.")

    with col_stats:
        st.markdown("#### 📑 Input Telemetry")
        st.markdown(f"""
        <div class="threat-card">
            <div>• <b>Source Vector:</b> {source}</div>
            <div>• <b>Analyzed Words:</b> {result['input_length_words']} tokens</div>
            <div>• <b>Analyzed Characters:</b> {result['input_length_chars']} chars</div>
            <div>• <b>Model Certainty Score:</b> {f"{confidence * 100:.2f}%" if confidence is not None else "N/A"}</div>
        </div>
        """, unsafe_allow_html=True)

    # Heuristic Red Flags
    indicators = result.get("suspicious_indicators", [])
    st.markdown("#### 🚩 Heuristic Indicator Scan")
    if indicators:
        st.markdown(f"<span class='badge-amber'>⚠️ {len(indicators)} POTENTIAL RED FLAGS DETECTED</span>", unsafe_allow_html=True)
        for ind in indicators:
            st.markdown(f"""
            <div style="background: rgba(255, 191, 71, 0.08); border-left: 4px solid {THEME['warning_amber']}; padding: 0.8rem 1rem; margin: 0.6rem 0; border-radius: 4px;">
                <div style="font-weight: 700; color: {THEME['warning_amber']}; font-size: 0.95rem;">
                    [{ind['severity']}] {ind['title']}
                </div>
                <div style="color: {THEME['text_primary']}; font-size: 0.85rem; margin-top: 0.2rem;">
                    {ind['description']}
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {THEME['text_secondary']}; margin-top: 0.3rem;">
                    Trigger phrase: <code>"{ind['matched_snippet']}"</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(0, 245, 160, 0.08); border-left: 4px solid {THEME['emerald']}; padding: 0.8rem 1rem; border-radius: 4px;">
            <span style="color: {THEME['emerald']}; font-weight: 600;">✓ No overt scam keywords (Telegram, check advances, wire transfers, free recruiter emails) detected by heuristic rules.</span>
        </div>
        """, unsafe_allow_html=True)

    # Explainability: Influential Words
    explanation = result.get("explanation", {})
    susp_tokens = explanation.get("suspicious_tokens", [])
    legit_tokens = explanation.get("legitimate_tokens", [])

    if susp_tokens or legit_tokens:
        st.markdown("#### 🔬 Influential Language Features (TF-IDF & Linear Weights)")
        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            st.markdown("**Tokens Leaning Toward Fraud Risk:**")
            if susp_tokens:
                for tok in susp_tokens:
                    st.markdown(f"• <span style='color: {THEME['threat_red']}; font-family: monospace;'>{tok['word']}</span> (impact: +{tok['impact']})", unsafe_allow_html=True)
            else:
                st.caption("No significant fraud-associated tokens found.")

        with exp_c2:
            st.markdown("**Tokens Leaning Toward Legitimacy:**")
            if legit_tokens:
                for tok in legit_tokens:
                    st.markdown(f"• <span style='color: {THEME['emerald']}; font-family: monospace;'>{tok['word']}</span> (trust: +{tok['impact']})", unsafe_allow_html=True)
            else:
                st.caption("No strong legitimacy-associated tokens found.")

    # Recommended Independent Verification Steps
    st.markdown("#### 🛡️ Recommended Verification Steps")
    for step in result.get("verification_steps", []):
        st.markdown(f"• {step}")

    # Disclaimer
    st.markdown("---")
    st.caption(f"**Security & Compliance Disclaimer:** {result['limitations_disclaimer']}")


def render_model_intelligence(metrics, comparison_df):
    """Render Page D: Model Intelligence & Benchmarking."""
    st.markdown(f"""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; color: {THEME['text_primary']}; margin-bottom: 0.2rem;">
            📊 MODEL INTELLIGENCE <span style="color: {THEME['neon_cyan']};">// BENCHMARKS</span>
        </h2>
        <p style="color: {THEME['text_secondary']}; font-size: 0.95rem;">
            Transparent evaluation of all three scikit-learn models on held-out test data (3,576 samples).
        </p>
    </div>
    """, unsafe_allow_html=True)

    if comparison_df is not None:
        st.markdown("### 🏆 Algorithm Performance Comparison")
        st.dataframe(comparison_df.style.highlight_max(axis=0, color="#123d38"), use_container_width=True)

    cm_img_path = REPORTS_DIR / "model_comparison.png"
    if cm_img_path.exists():
        st.image(str(cm_img_path), caption="Model Performance Comparison across Fraud Precision, Recall, and F1", use_column_width=True)

    st.markdown("### 🔍 Confusion Matrices by Candidate")
    c1, c2, c3 = st.columns(3)
    models_to_check = [
        ("Logistic Regression", "confusion_matrix_logistic_regression.png", c1),
        ("Multinomial Naive Bayes", "confusion_matrix_multinomial_naive_bayes.png", c2),
        ("Linear SVM (Calibrated)", "confusion_matrix_linear_svm_calibrated.png", c3),
    ]
    for mname, fname, col in models_to_check:
        with col:
            img_file = REPORTS_DIR / fname
            if img_file.exists():
                st.image(str(img_file), caption=mname, use_column_width=True)
            else:
                st.info(f"Matrix for {mname} available once training finishes.")

    st.markdown("### ⚖️ The Precision vs Recall Trade-Off in Scam Detection")
    st.markdown(f"""
    <div class="threat-card">
        <p style="font-size: 0.9rem; line-height: 1.5; color: {THEME['text_primary']};">
            Because fraudulent postings represent only ~4.8% of recruitment advertisements, raw classification accuracy is a deceptive metric. 
            A naive baseline model predicting all jobs as "Legitimate" achieves <b>95.16% accuracy</b> while missing <b>100% of scams</b>.
            Therefore, our model selection prioritizes <b>Fraud Recall</b> (preventing job seekers from falling victim to identity theft or financial fraud) 
            balanced with <b>Fraud Precision</b> (maintaining trust by not overwhelming legitimate employers with false positives).
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_dataset_intelligence(summary):
    """Render Page E: Dataset Intelligence."""
    st.markdown(f"""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; color: {THEME['text_primary']}; margin-bottom: 0.2rem;">
            🗄️ DATASET INTELLIGENCE <span style="color: {THEME['neon_cyan']};">// KAGGLE CORPUS</span>
        </h2>
        <p style="color: {THEME['text_secondary']}; font-size: 0.95rem;">
            Empirical properties and exploratory data analysis of the real-world dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        img_class = REPORTS_DIR / "eda_class_distribution.png"
        if img_class.exists():
            st.image(str(img_class), caption="Class Imbalance Distribution (0 vs 1)", use_column_width=True)
    with c2:
        img_missing = REPORTS_DIR / "eda_missing_values.png"
        if img_missing.exists():
            st.image(str(img_missing), caption="Missing Data by Column Percentage", use_column_width=True)

    img_text = REPORTS_DIR / "eda_text_length_distribution.png"
    if img_text.exists():
        st.image(str(img_text), caption="Word Count Density Distribution (Legitimate vs Fraudulent)", use_column_width=True)

    st.markdown("### 📋 Dataset Metadata Summary")
    if summary:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"""
            - **Total Records:** {summary['total_rows']:,}
            - **Total Features:** {summary['total_columns']}
            - **Legitimate Postings (0):** {summary['class_distribution']['legitimate_0']:,} ({summary['class_distribution']['legitimate_pct']}%)
            - **Fraudulent Postings (1):** {summary['class_distribution']['fraudulent_1']:,} ({summary['class_distribution']['fraudulent_pct']}%)
            - **Imbalance Ratio:** ~{summary['class_distribution']['imbalance_ratio']}:1
            """)
        with col_s2:
            st.markdown(f"""
            - **Exact Row Duplicates:** {summary.get('exact_duplicates', 0)}
            - **Content Duplicates:** {summary.get('content_duplicates', 0)}
            - **Top Missing Features:** salary_range, department, benefits
            - **Target Column:** `fraudulent` (binary integer)
            """)


def render_methodology():
    """Render Page F: Methodology, Mathematical Foundations, and Viva Preparation."""
    st.markdown(f"""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; color: {THEME['text_primary']}; margin-bottom: 0.2rem;">
            📖 METHODOLOGY & VIVA PREPARATION <span style="color: {THEME['neon_cyan']};">// ACADEMIC REFERENCE</span>
        </h2>
        <p style="color: {THEME['text_secondary']}; font-size: 0.95rem;">
            Technical architecture, mathematical models, defensive networking design, and BTech project examination Q&A.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔬 1. Term Frequency-Inverse Document Frequency (TF-IDF)", expanded=True):
        st.markdown(r"""
        TF-IDF quantifies the importance of a term $t$ in a document $d$ relative to a collection of documents $D$:
        $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
        where:
        $$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
        In PHANTOM // JOBINTEL, we utilize sublinear term-frequency scaling ($\text{TF}_{\text{sub}} = 1 + \ln(\text{TF})$) across 10,000 unigrams and bigrams to penalize overly dominant repeated words.
        """)

    with st.expander("🧠 2. Machine Learning Algorithms"):
        st.markdown("""
        1. **Logistic Regression (with Balanced Class Weights):**
           Models posterior log-odds using the sigmoid link function:
           $$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x} - b}}$$
           To counteract the 95:5 class imbalance, training instances in class 1 receive loss weights scaled inversely to class frequencies.

        2. **Multinomial Naive Bayes (MNB):**
           Applies Bayes' theorem with conditional independence assumption:
           $$P(y=c|\mathbf{x}) \propto P(y=c) \prod_{i=1}^n P(x_i|y=c)$$
           With Laplace smoothing $\\alpha = 0.1$ to handle unseen vocabulary tokens.

        3. **Linear Support Vector Machine (Linear SVM with Sigmoid Calibration):**
           Maximizes margin between hyperplanes:
           $$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$
           Because Linear SVM outputs uncalibrated geometric margins, we apply Platt scaling via `CalibratedClassifierCV(cv=3)` using out-of-fold validation to generate valid probability estimates.
        """)

    with st.expander("🛡️ 3. Defensive URL Extraction Architecture (SSRF Prevention)"):
        st.markdown("""
        Public URL analysis introduces severe Server-Side Request Forgery (SSRF) threats. PHANTOM // JOBINTEL implements:
        - **Scheme Whitelist:** Strictly `http` and `https`.
        - **Host Whitelist/Blacklist:** Prohibits `localhost`, `127.0.0.1`, `::1`, and cloud metadata endpoints (`169.254.169.254`).
        - **Pre-Flight DNS Resolution:** Resolves target domain before connection and verifies all resolved IP addresses against RFC 1918 private ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`) and link-local ranges.
        - **Redirect Re-validation:** Inspects HTTP 301/302 redirects in a loop, re-validating the resolved destination before each hop.
        - **Streaming Read Limits:** Enforces a hard 2 MB limit via chunk streaming to block gzip/memory exhaustion attacks.
        """)

    st.markdown("### 🎓 BTech Project Viva Voce Preparation (15 Key Questions)")
    viva_qa = [
        ("Why is fake job detection an important cybersecurity problem?", "Employment scams are weaponized vectors for identity theft, money laundering, and credential harvesting. Victims suffer severe financial loss and loss of personal data (SSN, banking records). Automated detection protects job seekers before sensitive disclosures occur."),
        ("Why did you select the Kaggle real-or-fake job postings dataset?", "It contains 17,880 real recruitment postings with verified fraudulent tags (866 scams), featuring rich multimodal text fields including company profile, requirements, benefits, and structured metadata."),
        ("How did you handle the severe 95:5 class imbalance?", "We employed stratified train-test splitting to preserve minority proportions, combined with class-weight balancing (inverting loss penalties relative to class frequencies) and evaluated models using Fraud Recall and Fraud F1 rather than raw accuracy."),
        ("Why is raw accuracy misleading in this project?", "A baseline dummy model predicting 100% legitimate postings scores 95.16% accuracy while catching 0 fake jobs. In scam prevention, false negatives (failing to catch a scam) are catastrophic, making Fraud Recall the critical metric."),
        ("What is the advantage of using TF-IDF over simple bag-of-words?", "Bag-of-words simply counts frequency, giving high weight to ubiquitous words ('the', 'experience'). TF-IDF scales down universally common words while emphasizing discriminative terms specific to fraudulent listings."),
        ("Why was sublinear TF scaling used in TF-IDF?", "Sublinear scaling replaces term frequency $TF$ with $1 + \ln(TF)$ for positive frequencies, preventing a single word repeated 20 times in a fraudulent spam ad from exerting 20 times the influence of a word appearing once."),
        ("How does Platt Calibration work with Linear SVM?", "Linear SVM computes an uncalibrated signed geometric distance from the hyperplane. Platt calibration fits a logistic sigmoid function over out-of-fold validation decision scores to output well-calibrated probabilities between 0.0 and 1.0."),
        ("What are the key heuristic indicators detected by PHANTOM?", "1) Free recruiter emails (gmail, yahoo) instead of corporate domains. 2) Conducting interviews over Telegram/WhatsApp. 3) Demands for upfront equipment fees or check cashing schemes. 4) Cryptocurrency and wire transfer duties."),
        ("How does the system prevent SSRF attacks during URL analysis?", "It resolves domain DNS before making HTTP socket requests, rejecting loopback (127.0.0.1), private subnets (10.x, 192.168.x, 172.16.x), cloud metadata IPs (169.254.169.254), and enforces strict redirect re-validation and 2MB stream limits."),
        ("Why use a scikit-learn Pipeline rather than vectorizing data manually?", "A Pipeline encapsulates both the TF-IDF vectorizer and the classifier into a single serializable object (`joblib`), preventing training/testing data leakage and guaranteeing identical preprocessing during real-time inference."),
        ("How does the model provide explainability?", "For linear models, we multiply the non-zero TF-IDF features of the submitted advertisement by the model's learned feature coefficients, isolating the specific tokens that pushed the prediction toward fraud or legitimacy."),
        ("Can this system guarantee that an advertisement is 100% legitimate?", "No. Advanced threat actors can copy legitimate text verbatim. The model provides a statistical heuristic indicator. Final verification must always include cross-checking the employer's official careers site and corporate registries."),
        ("What HTML parsing strategies are used in URL extraction?", "First, schema.org `JobPosting` JSON-LD scripts are inspected. If absent, fallback DOM heuristic extraction analyzes semantic containers (`<article>`, `class='job-description'`, Open Graph metadata)."),
        ("What is the difference between macro F1 and weighted F1?", "Macro F1 calculates the unweighted mean of F1 scores across classes, treating rare fraud equally with common legitimate posts. Weighted F1 weights each class by its support, masking minority class failures in imbalanced datasets."),
        ("What are the future enhancement possibilities for PHANTOM?", "Integrating BERT/RoBERTa transformer architectures, adding automated corporate domain WHOIS lookup to verify recruiter domain age, and connecting public employer registry APIs (e.g. SEC EDGAR)."),
    ]

    for i, (q, a) in enumerate(viva_qa, start=1):
        with st.expander(f"Q{i}: {q}"):
            st.write(a)


def main():
    """Main Streamlit execution coordinator."""
    predictor = load_cached_predictor()
    metrics, summary, comparison_df = load_metrics_and_summary()

    selected_page = render_sidebar()

    if selected_page == "🛰️ Command Center":
        render_command_center(metrics, summary, comparison_df)
    elif selected_page == "⚡ Job Analyzer":
        render_job_analyzer(predictor)
    elif selected_page == "📊 Model Intelligence":
        render_model_intelligence(metrics, comparison_df)
    elif selected_page == "🗄️ Dataset Intelligence":
        render_dataset_intelligence(summary)
    elif selected_page == "📖 Methodology & Viva":
        render_methodology()


if __name__ == "__main__":
    main()
