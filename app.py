
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

from scipy.sparse import hstack, csr_matrix
import time
import html
import sys
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parent)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from llm_engine import generate_response


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dynamic Model Selection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #172554 0%, transparent 35%),
        radial-gradient(circle at 90% 20%, #312e81 0%, transparent 30%),
        #070b16;
    color: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 32px;
    border-radius: 24px;
    background: linear-gradient(135deg, #111936, #172554);
    border: 1px solid #334477;
    box-shadow: 0 15px 50px rgba(0,0,0,.30);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 850;
}

.hero-subtitle {
    font-size: 17px;
    color: #aab7d8;
    margin-top: 6px;
}

.status {
    display: inline-block;
    padding: 8px 15px;
    border-radius: 30px;
    background: #123c2a;
    color: #65e6a3;
    font-weight: 700;
    font-size: 13px;
}

.metric-card {
    background: rgba(17, 26, 49, .92);
    border: 1px solid #2c3c66;
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    min-height: 120px;
}

.metric-value {
    font-size: 29px;
    font-weight: 850;
}

.metric-label {
    color: #9eabd0;
    margin-top: 6px;
    font-size: 14px;
}

.result-card {
    margin-top: 22px;
    padding: 32px;
    border-radius: 22px;
    background: linear-gradient(135deg, #111a35, #182443);
    border: 1px solid #3b4d7a;
    text-align: center;
}

.result-label {
    color: #9eabd0;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 13px;
}

.result-value {
    font-size: 46px;
    font-weight: 900;
    margin: 8px;
}

.low {
    color: #4ade80;
}

.medium {
    color: #facc15;
}

.high {
    color: #fb7185;
}

.info-card {
    background: #10182d;
    border: 1px solid #2b3b65;
    border-radius: 17px;
    padding: 20px;
    margin-top: 15px;
}

.pipeline-box {
    background: #10182d;
    border: 1px solid #334477;
    border-radius: 14px;
    padding: 18px 10px;
    text-align: center;
    font-weight: 750;
}

div.stButton > button {
    border-radius: 12px;
    height: 48px;
    font-weight: 750;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD FINAL OFFLINE MODEL
# ============================================================

@st.cache_resource
def load_models():

    classifier = joblib.load(
        PROJECT_DIR + "/final_complexity_classifier.pkl"
    )

    vectorizer = joblib.load(
        PROJECT_DIR + "/final_tfidf_vectorizer.pkl"
    )

    scaler = joblib.load(
        PROJECT_DIR + "/final_complexity_scaler.pkl"
    )

    numeric_scaler = joblib.load(
        PROJECT_DIR + "/final_numeric_scaler.pkl"
    )

    encoder = joblib.load(
        PROJECT_DIR + "/final_metadata_encoder.pkl"
    )

    return classifier, vectorizer, scaler, encoder, numeric_scaler


try:

    classifier, vectorizer, scaler, encoder, numeric_scaler = load_models()

    MODEL_READY = True

except Exception as e:

    MODEL_READY = False
    st.error(f"Model loading error: {e}")

# ============================================================
# LATENCY OPTIMIZATION BENCHMARK
# ============================================================

# Controlled 3-query benchmark using warmed local LLM models.
# Measures LLM generation time only.
# Adaptive routing vs always using the Large model.

LATENCY_BENCHMARK = {
    "Adaptive Routing": 27.0600,
    "Always-Large Baseline": 36.9460
}

LATENCY_TIME_SAVED = (
    LATENCY_BENCHMARK["Always-Large Baseline"]
    - LATENCY_BENCHMARK["Adaptive Routing"]
)

LATENCY_REDUCTION = (
    LATENCY_TIME_SAVED
    / LATENCY_BENCHMARK["Always-Large Baseline"]
) * 100

LATENCY_QUERY_RESULTS = pd.DataFrame({
    "Query Complexity": [
        "LOW",
        "MEDIUM",
        "HIGH"
    ],
    "Adaptive Model": [
        "Qwen 0.5B",
        "Qwen 1.5B",
        "Qwen 3B"
    ],
    "Adaptive LLM Time (s)": [
        7.5910,
        7.1607,
        12.3084
    ],
    "Always-Large Time (s)": [
        12.3275,
        12.3005,
        12.3180
    ]
})



# ============================================================
# FEATURE EXTRACTION
# ============================================================


def extract_original_features(text):

    words = text.split()

    return [
        len(words),
        len(text),
        max(1, sum(text.count(x) for x in [".", "!", "?"])),
        np.mean([len(w) for w in words]) if words else 0,
        sum(char.isdigit() for char in text),
        sum(char.isupper() for char in text),
        sum(
            not char.isalnum() and not char.isspace()
            for char in text
        ),
        text.count("?"),
        sum(
            text.count(symbol)
            for symbol in ["+", "-", "*", "/", "=", "<", ">", "^"]
        )
    ]


def extract_complexity_features(text):

    words = text.split()

    word_count = len(words)

    char_count = len(text)

    sentence_count = sum(
        text.count(x) for x in [".", "!", "?"]
    )

    avg_word_length = (
        np.mean([len(w) for w in words])
        if words else 0
    )

    unique_word_ratio = (
        len(set(text.lower().split())) / max(len(words), 1)
    )

    digit_count = sum(
        c.isdigit() for c in text
    )

    uppercase_count = sum(
        c.isupper() for c in text
    )

    special_char_count = sum(
        not c.isalnum() and not c.isspace()
        for c in text
    )

    question_mark_count = text.count("?")

    code_symbol_count = len(
        re.findall(r"[{}\[\]();=<>]", text)
    )

    math_symbol_count = len(
        re.findall(r"[+\-*/^=]", text)
    )

    newline_count = text.count("\n")

    has_code = int(
        bool(
            re.search(
                r"\b(function|class|SELECT|INSERT|UPDATE|import|def|public|private|return)\b",
                text,
                re.IGNORECASE
            )
        )
    )

    has_math = int(
        bool(
            re.search(
                r"[=+\-*/^]|\\frac|\\sum|\\int|proof|theorem|equation",
                text,
                re.IGNORECASE
            )
        )
    )

    return [
        char_count,
        word_count,
        sentence_count,
        avg_word_length,
        unique_word_ratio,
        digit_count,
        uppercase_count,
        special_char_count,
        question_mark_count,
        code_symbol_count,
        math_symbol_count,
        newline_count,
        has_code,
        has_math
    ]


# PREDICTION PIPELINE

# ============================================================

def predict_query(text):

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    tfidf = vectorizer.transform([text])


    # --------------------------------------------------------
    # ORIGINAL 9 NUMERICAL FEATURES
    # --------------------------------------------------------

    original_numeric = np.array(
        [extract_original_features(text)]
    )

    original_numeric = numeric_scaler.transform(original_numeric)

    base_features = hstack([
        tfidf,
        csr_matrix(original_numeric)
    ])


    # --------------------------------------------------------
    # 14 COMPLEXITY FEATURES
    # --------------------------------------------------------

    complexity_values = extract_complexity_features(text)

    complexity_columns = [
        "char_count",
        "word_count",
        "sentence_count",
        "avg_word_length",
        "unique_word_ratio",
        "digit_count",
        "uppercase_count",
        "special_char_count",
        "question_mark_count",
        "code_symbol_count",
        "math_symbol_count",
        "newline_count",
        "has_code",
        "has_math"
    ]

    complexity_df = pd.DataFrame(
        [complexity_values],
        columns=complexity_columns
    )

    scaled_complexity = scaler.transform(
        complexity_df
    )


    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = pd.DataFrame({
        "subject": ["UNKNOWN"],
        "source": ["UNKNOWN"]
    })

    metadata_features = encoder.transform(
        metadata
    )


    # --------------------------------------------------------
    # FINAL 6562 FEATURE VECTOR
    # --------------------------------------------------------

    final_features = hstack([
        base_features,
        csr_matrix(scaled_complexity),
        metadata_features
    ])


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    prediction = classifier.predict(
        final_features
    )[0]


    # --------------------------------------------------------
    # DECISION MARGIN
    # --------------------------------------------------------

    scores = classifier.decision_function(
        final_features
    )[0]

    sorted_scores = np.sort(scores)

    margin = (
        sorted_scores[-1] -
        sorted_scores[-2]
    )


    # --------------------------------------------------------
    # ROUTING POLICY
    # --------------------------------------------------------

    routing = {

        "LOW": {
            "model": "SMALL MODEL",
            "cost": 1.0,
            "message":
                "Simple query detected → routed to the LOW-cost model."
        },

        "MEDIUM": {
            "model": "MEDIUM MODEL",
            "cost": 2.0,
            "message":
                "Moderate complexity detected → routed to the MEDIUM model."
        },

        "HIGH": {
            "model": "LARGE MODEL",
            "cost": 4.0,
            "message":
                "High complexity detected → routed to the HIGH-capability model."
        }
    }

    route = routing[prediction]


    return {
        "complexity": prediction,
        "margin": margin,
        "model": route["model"],
        "cost": route["cost"],
        "message": route["message"],
        "feature_count": final_features.shape[1],
        "original_features": original_numeric[0],
        "complexity_features": complexity_values
    }


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div style="display:flex; justify-content:space-between;
align-items:center;">

<div>

<div class="hero-title">
⚡ Dynamic Model Selection
</div>

<div class="hero-subtitle">
Adaptive LLM Routing for Cost-Efficient Model Selection
</div>

</div>

<div class="status">
● OFFLINE SYSTEM READY
</div>

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚡ SYSTEM")

    if MODEL_READY:
        st.success("Classifier Loaded")
        st.success("TF-IDF Loaded")
        st.success("Feature Pipeline Ready")
    else:
        st.error("Model Loading Failed")

    st.markdown("---")

    st.markdown("### Model Tiers")

    st.markdown("🟢 **SMALL** — 1.0x")
    st.markdown("🟡 **MEDIUM** — 2.0x")
    st.markdown("🔴 **LARGE** — 4.0x")

    st.markdown("---")

    st.markdown("### Project")

    st.write("Dataset: **3,750 queries**")
    st.write("Classes: **3**")
    st.write("Features: **6,562**")
    st.write("Classifier: **Linear SVM**")

    st.markdown("---")

    st.caption(
        "100% offline inference • No paid API required"
    )



# ============================================================
# TOP PERFORMANCE CARDS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-value">67.33%</div>
    <div class="metric-label">Routing Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-value">65.97%</div>
    <div class="metric-label">Macro F1</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-value">40.43%</div>
    <div class="metric-label">Normalized Cost Saving</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-value">26.76%</div>
    <div class="metric-label">LLM Latency Reduction</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-label">LLM Generation Time</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# LATENCY PROOF BANNER
# ------------------------------------------------------------

st.markdown("""
<div class="info-card"
     style="
        border:1px solid #3156a3;
        background:linear-gradient(135deg,#101a36,#162852);
        margin-top:20px;
        margin-bottom:20px;
     ">

<h3 style="margin-bottom:8px;">
</h3>

<div style="
    font-size:18px;
    color:#f8fafc;
    margin-bottom:8px;
">
Adaptive Routing:
<b>27.06 seconds</b>
&nbsp;&nbsp;→&nbsp;&nbsp;
Always-Large:
<b>36.95 seconds</b>
</div>

<div style="
    font-size:28px;
    font-weight:900;
    color:#65e6a3;
">
</div>

<div style="
    color:#aab7d8;
    margin-top:8px;
">
Measured on the same 3 representative queries using warmed models.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

# ============================================================

tab_router, tab_analytics, tab_pipeline, tab_about = st.tabs([
    "⚡ LIVE ROUTER",
    "📊 ANALYTICS",
    "🧠 HOW IT WORKS",
    "ℹ️ ABOUT"
])


# ============================================================
# LIVE ROUTER
# ============================================================

with tab_router:

    st.markdown("## Ask the Adaptive Router")

    st.caption(
        "Enter a query and the trained offline classifier "
        "will determine its complexity and model tier."
    )

    # Demo buttons
    d1, d2, d3, d4 = st.columns(4)

    if "query" not in st.session_state:
        st.session_state.query = ""

    if d1.button(
        "💡 Simple Query",
        use_container_width=True
    ):
        st.session_state.query = (
            "What is the capital of France?"
        )

    if d2.button(
        "💻 Coding Task",
        use_container_width=True
    ):
        st.session_state.query = (
            "Implement a Python function to calculate Fibonacci numbers."
        )

    if d3.button(
        "🧮 Reasoning Task",
        use_container_width=True
    ):
        st.session_state.query = (
            "Prove that A^T A is symmetric and positive semidefinite."
        )

    if d4.button(
        "🧹 Clear",
        use_container_width=True
    ):
        st.session_state.query = ""


    query = st.text_area(
        "Query",
        value=st.session_state.query,
        height=130,
        placeholder=(
            "Example: Explain the difference between "
            "TCP and UDP..."
        )
    )


    if st.button(
        "⚡ ANALYZE & ROUTE QUERY",
        type="primary",
        use_container_width=True
    ):

        if not query.strip():

            st.warning(
                "Please enter a query."
            )

        elif not MODEL_READY:

            st.error(
                "Final model files could not be loaded."
            )

        else:

            result = predict_query(
                query.strip()
            )

            # ====================================================
            # ACTUAL LOCAL LLM GENERATION
            # ====================================================

            total_start = time.perf_counter()

            try:
                # ========================================================
                # ADAPTIVE ROUTING — ACTUAL SELECTED MODEL
                # ========================================================

                llm_result = generate_response(
                    query.strip(),
                    result["complexity"]
                )

                result["llm_response"] = llm_result["response"]
                result["llm_model"] = llm_result["model_name"]
                result["llm_latency"] = llm_result["llm_latency"]
                result["input_tokens"] = llm_result["input_tokens"]
                result["output_tokens"] = llm_result["output_tokens"]
                result["total_tokens"] = llm_result["total_tokens"]
                result["api_cost"] = 0.0

                # ========================================================
                # ALWAYS-LARGE BASELINE — SAME EXACT QUERY
                # ========================================================
                #
                # This measures what would happen if every query were
                # routed to the Large Qwen model.
                #
                # We compare LLM GENERATION TIME only.
                # Model-loading overhead is not included in llm_latency.
                # ========================================================

                always_large_result = generate_response(
                    query.strip(),
                    "HIGH"
                )

                result["always_large_model"] = (
                    always_large_result["model_name"]
                )

                result["always_large_latency"] = float(
                    always_large_result["llm_latency"]
                )

                # ========================================================
                # QUERY-SPECIFIC LATENCY OPTIMIZATION
                # ========================================================

                routed_time = float(result["llm_latency"])
                always_large_time = float(
                    result["always_large_latency"]
                )

                result["latency_time_saved"] = (
                    always_large_time - routed_time
                )

                if always_large_time > 0:
                    result["latency_reduction"] = (
                        (
                            always_large_time - routed_time
                        ) / always_large_time
                    ) * 100
                else:
                    result["latency_reduction"] = 0.0

            except Exception as llm_error:
                result["llm_error"] = str(llm_error)

            result["end_to_end_latency"] = (
                time.perf_counter() - total_start
            )

            complexity = result["complexity"]

            st.markdown(
                f"""
                <div class="result-card">

                <div class="result-label">
                Predicted Query Complexity
                </div>

                <div class="result-value {complexity.lower()}">
                {complexity}
                </div>

                <div style="color:#aab7d8;">
                Decision Margin:
                <b>{result["margin"]:.4f}</b>
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                    <div class="metric-value">
                    {result["model"]}
                    </div>
                    <div class="metric-label">
                    Selected Model Tier
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with r2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                    <div class="metric-value">
                    {result["cost"]:.1f}x
                    </div>
                    <div class="metric-label">
                    Relative Routing Cost
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with r3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                    <div class="metric-value">
                    {result["feature_count"]}
                    </div>
                    <div class="metric-label">
                    Final Feature Vector
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.markdown(
                f"""
                <div class="info-card">

                <h4>🧭 Routing Decision</h4>

                <p>{result["message"]}</p>

                </div>
                """,
                unsafe_allow_html=True
            )



            # ====================================================
            # ACTUAL LLM OUTPUT + PERFORMANCE
            # ====================================================

            if "llm_response" in result:

                st.markdown("## 🤖 Final LLM Output")

                safe_response = html.escape(
                    str(result["llm_response"])
                )

                st.markdown(
                    f"""
                    <div class="info-card"
                         style="text-align:left;
                                line-height:1.65;
                                font-size:16px;">
                    {safe_response}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("## ⏱️ Actual LLM Performance")

                t1, t2, t3, t4 = st.columns(4)

                with t1:
                    st.metric(
                        "LLM Generation Time",
                        f'{result["llm_latency"]:.3f} s'
                    )

                with t2:
                    st.metric(
                        "End-to-End Time",
                        f'{result["end_to_end_latency"]:.3f} s'
                    )

                with t3:
                    st.metric(
                        "Output Tokens",
                        int(result["output_tokens"])
                    )

                with t4:
                    st.metric(
                        "Total Tokens",
                        int(result["total_tokens"])
                    )

                st.caption(
                    "LLM generation time is measured locally on the GPU. "
                    "End-to-end time includes routing and LLM execution."
                )

                # ========================================================
                # QUERY-LEVEL LATENCY OPTIMIZATION PROOF
                # ========================================================

                routed_time = float(result["llm_latency"])

                always_large_time = float(
                    result["always_large_latency"]
                )

                time_saved = float(
                    result["latency_time_saved"]
                )

                latency_reduction = float(
                    result["latency_reduction"]
                )

                # --------------------------------------------------------
                # QUERY-SPECIFIC LATENCY COMPARISON
                # --------------------------------------------------------

                st.markdown("### ⚡ Latency Optimization for This Query")

                st.caption(
                    "Same query measured on the routed model and the "
                    "Always-Large baseline."
                )

                p1, p2, p3 = st.columns(3)

                with p1:
                    st.metric(
                        "Routed Model LLM Time",
                        f"{routed_time:.3f} s"
                    )
                    st.caption(
                        str(result["llm_model"])
                    )

                with p2:
                    st.metric(
                        "Always-Large LLM Time",
                        f"{always_large_time:.3f} s"
                    )
                    st.caption(
                        str(result["always_large_model"])
                    )

                with p3:
                    st.metric(
                        "Time Difference",
                        f"{time_saved:.3f} s"
                    )
                    st.caption(
                        "Routed − Always-Large"
                    )

                # --------------------------------------------------------
                # QUERY-SPECIFIC PERCENTAGE
                # --------------------------------------------------------

                if latency_reduction >= 0:

                    st.success(
                        f"⚡ {latency_reduction:.2f}% "
                        f"LLM Generation Latency Reduction"
                    )

                else:

                    st.warning(
                        f"⚠️ {abs(latency_reduction):.2f}% "
                        f"LLM Generation Latency Increase "
                        f"for this query"
                    )

                st.caption(
                    f"Calculation: "
                    f"({always_large_time:.3f} − "
                    f"{routed_time:.3f}) ÷ "
                    f"{always_large_time:.3f} × 100 "
                    f"= {latency_reduction:.2f}%"
                )

                st.markdown("### 💰 Actual Cost / Usage")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Normalized Routing Cost",
                        f'{result["cost"]:.1f}x'
                    )

                with c2:
                    st.metric(
                        "Actual API Expenditure",
                        "₹0"
                    )

                with c3:
                    st.metric(
                        "Input Tokens",
                        int(result["input_tokens"])
                    )


                st.markdown("### 🧠 Model Actually Used")

                st.markdown(
                    f"""
                    <div class="pipeline-box">
                {html.escape(str(result["llm_model"]))}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif "llm_error" in result:

                st.error(
                    "LLM generation failed: "
                    + str(result["llm_error"])
                )

            st.markdown("### 🔍 Query Characteristics")

            # Calculate display statistics directly from the submitted query
            features = extract_original_features(query)

            q1, q2, q3, q4, q5 = st.columns(5)

            q1.metric(
                "Words",
                int(features[0])
            )

            q2.metric(
                "Characters",
                int(features[1])
            )

            q3.metric(
                "Sentences",
                int(features[2])
            )

            q4.metric(
                "Digits",
                int(features[4])
            )

            q5.metric(
                "Special Characters",
                int(features[6])
            )


            st.markdown("### 🔄 Live Routing Path")

            p1, p2, p3, p4 = st.columns(4)

            with p1:
                st.markdown(
                    '<div class="pipeline-box">USER QUERY</div>',
                    unsafe_allow_html=True
                )

            with p2:
                st.markdown(
                    '<div class="pipeline-box">NLP + FEATURES</div>',
                    unsafe_allow_html=True
                )

            with p3:
                st.markdown(
                    '<div class="pipeline-box">SVM CLASSIFIER</div>',
                    unsafe_allow_html=True
                )

            with p4:
                st.markdown(
                    f'<div class="pipeline-box">{complexity} → {result["model"]}</div>',
                    unsafe_allow_html=True
                )


# ============================================================
# ANALYTICS
# ============================================================

with tab_analytics:

    # ========================================================
    # LATENCY OPTIMIZATION PROOF
    # ========================================================

    st.markdown("---")
    st.markdown("## ⚡ Latency Optimization Proof")

    st.markdown("""
    <div class="info-card">

    <h3>Adaptive Routing vs Always-Large</h3>

    <p style="color:#aab7d8;font-size:15px;">
    The same three representative queries were evaluated using:
    <b>Adaptive Routing</b> and an
    <b>Always-Large baseline</b>.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # MAIN LATENCY METRICS
    # --------------------------------------------------------

    l1, l2, l3, l4 = st.columns(4)

    with l1:
        st.metric(
            "Adaptive Routing",
            "27.06 s"
        )

    with l2:
        st.metric(
            "Always-Large",
            "36.95 s"
        )

    with l3:
        st.metric(
            "Time Saved",
            "9.89 s"
        )

    with l4:
        st.metric(
            "Latency Reduction",
            "26.76%"
        )

    # --------------------------------------------------------
    # LATENCY COMPARISON CHART
    # --------------------------------------------------------

    st.markdown("### 📉 LLM Generation Time Comparison")

    latency_chart_df = pd.DataFrame({
        "Strategy": [
            "Adaptive Routing",
            "Always-Large"
        ],
        "LLM Generation Time (seconds)": [
            27.0600,
            36.9460
        ]
    })

    st.bar_chart(
        latency_chart_df.set_index("Strategy")
    )

    st.caption(
        "Lower generation time is better. "
        "This benchmark measures LLM generation time only."
    )

    # --------------------------------------------------------
    # QUERY-LEVEL PROOF
    # --------------------------------------------------------

    st.markdown("### 🔬 Query-Level Latency Evidence")

    display_latency = LATENCY_QUERY_RESULTS.copy()

    display_latency["Adaptive LLM Time (s)"] = (
        display_latency["Adaptive LLM Time (s)"]
        .map(lambda x: f"{x:.3f}")
    )

    display_latency["Always-Large Time (s)"] = (
        display_latency["Always-Large Time (s)"]
        .map(lambda x: f"{x:.3f}")
    )

    st.dataframe(
        display_latency,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CALCULATION PROOF
    # --------------------------------------------------------

    st.markdown("### 🧮 How the 26.76% Was Calculated")

    st.markdown("""
    <div class="info-card">

    <div style="font-size:18px;line-height:1.8;">

    <b>Always-Large generation time</b>
    = 36.9460 s

    <br>

    <b>Adaptive generation time</b>
    = 27.0600 s

    <br>

    <b>Time saved</b>
    = 36.9460 − 27.0600
    = <b>9.8860 s</b>

    <br>

    <b>Latency reduction</b>
    =
    (9.8860 / 36.9460) × 100
    =
    <b style="color:#65e6a3;">26.76%</b>

    </div>

    </div>
    """, unsafe_allow_html=True)

    st.success(
        "✓ Adaptive routing reduced measured LLM generation time "
        "by 26.76% in the controlled 3-query benchmark."
    )



    st.markdown("## 📊 Model & Routing Analytics")

    st.markdown("### Model Comparison")

    model_results = pd.DataFrame({
        "Experiment": [
            "Baseline Logistic Regression",
            "Tuned Logistic Regression",
            "Best Linear SVM",
            "Enhanced SVM"
        ],
        "Accuracy (%)": [
            64.1333,
            65.4667,
            65.2000,
            67.3333
        ],
        "Macro F1 (%)": [
            59.2374,
            64.1517,
            63.7077,
            65.9719
        ]
    })

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("### 💰 Cost Optimization")

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Always Large",
        "3000"
    )

    a2.metric(
        "Adaptive Routing",
        "1787",
        "-1213"
    )

    a3.metric(
        "Cost Saving",
        "40.43%"
    )


    cost_df = pd.DataFrame({
        "Strategy": [
            "Always Large",
            "Adaptive Routing"
        ],
        "Cost": [
            3000,
            1787
        ]
    })

    st.bar_chart(
        cost_df.set_index("Strategy")
    )


    st.markdown("### 🎯 Per-Class Routing Accuracy")

    class_df = pd.DataFrame({
        "Complexity": [
            "LOW",
            "MEDIUM",
            "HIGH"
        ],
        "Accuracy": [
            72.8,
            40.8,
            88.4
        ]
    })

    st.bar_chart(
        class_df.set_index("Complexity")
    )


    st.markdown("### Confusion Matrix")

    confusion = pd.DataFrame(
        [
            [182, 49, 19],
            [100, 102, 48],
            [7, 22, 221]
        ],
        index=[
            "Actual LOW",
            "Actual MEDIUM",
            "Actual HIGH"
        ],
        columns=[
            "Pred LOW",
            "Pred MEDIUM",
            "Pred HIGH"
        ]
    )

    st.dataframe(
        confusion,
        use_container_width=True
    )


# ============================================================
# HOW IT WORKS
# ============================================================

with tab_pipeline:

    st.markdown("## 🧠 How the Adaptive Router Works")

    st.markdown(
        "Each incoming query is converted into a hybrid feature "
        "representation before the routing decision is made."
    )

    pipeline_steps = [

        ("1", "USER QUERY",
         "Natural-language query entered by the user."),

        ("2", "NLP / TEXT ANALYSIS",
         "Text and structural characteristics are extracted."),

        ("3", "TF-IDF",
         "6,508 textual features."),

        ("4", "COMPLEXITY FEATURES",
         "14 handcrafted complexity features."),

        ("5", "METADATA",
         "31 encoded metadata features."),

        ("6", "FEATURE FUSION",
         "Final 6,562-dimensional representation."),

        ("7", "LINEAR SVM",
         "Predicts LOW, MEDIUM or HIGH complexity."),

        ("8", "ADAPTIVE ROUTER",
         "Selects the corresponding model tier.")
    ]


    for number, title, description in pipeline_steps:

        st.markdown(
            f"""
            <div class="info-card">

            <h4>{number}. {title}</h4>

            <div style="color:#aab7d8;">
            {description}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("### Model Routing Policy")

    policy = pd.DataFrame({
        "Complexity": [
            "LOW",
            "MEDIUM",
            "HIGH"
        ],
        "Model Tier": [
            "SMALL MODEL",
            "MEDIUM MODEL",
            "LARGE MODEL"
        ],
        "Relative Cost": [
            1.0,
            2.0,
            4.0
        ]
    })

    st.dataframe(
        policy,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT
# ============================================================

with tab_about:

    st.markdown("## ℹ️ About Dynamic Model Selection")

    st.markdown("""
    ### Project Objective

    Build an adaptive routing framework that predicts the
    complexity of incoming queries and selects an appropriate
    model tier instead of always using the largest model.

    ### Final System

    **User Query → Feature Engineering → Linear SVM →
    Complexity Class → Adaptive Routing → Model Tier**

    ### Final Results

    **Routing Accuracy:** 67.33%

    **Macro F1:** 65.97%

    **Normalized Cost Saving:** 40.43%

    **Dataset:** 3,750 queries

    **Final Feature Representation:** 6,562 features

    ### Deployment

    The current demonstration performs inference locally using
    the trained model artifacts. No paid API is required.
    """)

    st.success(
        "✓ Offline inference • ✓ Trained model loaded • ✓ No paid API required"
    )
