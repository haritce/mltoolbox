"""
ML Toolbox — Streamlit Web Interface
A modular AI agent system for complete ML pipelines.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys
import io
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from pipeline import MLPipeline
from data.example_datasets import EXAMPLE_DATASETS, load_example

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Toolbox — Modular AI Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid rgba(99, 179, 237, 0.3);
    }
    .main-header h1 {
        color: #63b3ed;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Agent cards */
    .agent-card {
        background: linear-gradient(135deg, #1a202c, #2d3748);
        border: 1px solid #4a5568;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.4rem 0;
        border-left: 4px solid #63b3ed;
        transition: all 0.2s;
    }
    .agent-card:hover { border-left-color: #90cdf4; }
    .agent-card h4 { color: #63b3ed; margin: 0 0 0.2rem 0; font-size: 0.9rem; }
    .agent-card p { color: #a0aec0; margin: 0; font-size: 0.8rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a202c, #2d3748);
        border: 1px solid #4a5568;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-label { color: #a0aec0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #68d391; font-size: 2rem; font-weight: 800; }
    .metric-model { color: #63b3ed; font-size: 0.85rem; margin-top: 0.3rem; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #2d3748, #1a202c);
        border-left: 4px solid #63b3ed;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0 0.5rem 0;
    }
    .section-header h3 { color: #e2e8f0; margin: 0; font-size: 1.1rem; }

    /* Status badges */
    .badge-success { background: #276749; color: #9ae6b4; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
    .badge-warning { background: #744210; color: #fbd38d; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
    .badge-info { background: #2a4365; color: #90cdf4; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }

    /* Override streamlit defaults */
    .stButton > button {
        background: linear-gradient(135deg, #3182ce, #2b6cb0);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #63b3ed, #3182ce);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(49, 130, 206, 0.4);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #a0aec0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #63b3ed;
        border-bottom-color: #63b3ed;
    }

    /* Info boxes */
    .rec-box {
        background: rgba(45, 55, 72, 0.6);
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'running' not in st.session_state:
    st.session_state.running = False


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 ML Toolbox — Modular AI Agent System</h1>
    <p>Complete Machine Learning Pipeline with 10 Specialized AI Agents</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")

    # Data Source
    st.markdown("**📂 Data Source**")
    data_source = st.radio("", ["Upload CSV", "Use Example Dataset"], label_visibility="collapsed")

    df = None
    target_col = None
    dataset_name = "Dataset"

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                dataset_name = uploaded_file.name.replace('.csv', '')
                st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
                target_col = st.selectbox("🎯 Target Column", df.columns.tolist())
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        example_name = st.selectbox("Choose dataset", list(EXAMPLE_DATASETS.keys()))
        if st.button("Load Dataset", type="secondary"):
            with st.spinner("Loading..."):
                df, target_col = load_example(example_name)
                dataset_name = example_name.split('(')[0].strip().replace('🌸 ', '').replace('🎗️ ', '').replace('🍷 ', '').replace('🔬 ', '').replace('🌙 ', '')
                st.session_state.df = df
                st.session_state.target_col = target_col
                st.success(f"✅ {df.shape[0]}×{df.shape[1]}")

        if st.session_state.df is not None:
            df = st.session_state.df
            target_col = st.session_state.target_col

    st.markdown("---")
    st.markdown("**🧠 Pipeline Settings**")

    task_type = st.selectbox("Task Type", ["classification", "regression"])
    hp_method = st.selectbox("HP Optimization Method", ["random", "grid", "bayesian"])
    hp_n_iter = st.slider("HP Search Iterations", 5, 50, 15)
    top_k = st.slider("Max Features to Select", 1, 30, 10)

    st.markdown("**🔧 Optional Steps**")
    run_explainability = st.checkbox("Run SHAP + LIME", value=True)
    run_hp = st.checkbox("Run HP Optimization", value=True)

    st.markdown("---")

    # Agent overview
    st.markdown("**🤖 Active Agents**")
    agents_info = [
        ("🔍", "Data Understanding", "EDA & profiling"),
        ("🔧", "Preprocessing", "Imputation & encoding"),
        ("⚙️", "Feature Engineering", "MI, RFE, Polynomial"),
        ("📦", "Small Dataset", "LOO-CV & Bootstrap"),
        ("🤖", "Model Training", "7 algorithms"),
        ("🔬", "Hyperparameter", "Grid/Random/Bayes"),
        ("📊", "Evaluation", "5+ metrics"),
        ("📈", "Visualization", "Plots & charts"),
        ("🧠", "Explainability", "SHAP + LIME"),
        ("📝", "Research Paper", "Auto-generation"),
    ]
    for icon, name, desc in agents_info:
        st.markdown(f"""
        <div class="agent-card">
            <h4>{icon} {name}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ─── Main Content ──────────────────────────────────────────────────────────────
if df is None:
    # Welcome screen
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Agents</div>
            <div class="metric-value">10</div>
            <div class="metric-model">Specialized AI Agents</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ML Models</div>
            <div class="metric-value">7</div>
            <div class="metric-model">Trained & Compared</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dataset Support</div>
            <div class="metric-value">∞</div>
            <div class="metric-model">Small to Large (n&lt;30 → n&gt;10k)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    st.info("👈 **Upload your CSV dataset** or **select an example dataset** from the sidebar to begin.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔄 Pipeline Flow")
        steps = [
            "1️⃣ Data Understanding → EDA, correlations, outliers",
            "2️⃣ Preprocessing → Imputation, encoding, scaling",
            "3️⃣ Small Dataset Opt. → LOO-CV, bootstrapping",
            "4️⃣ Feature Engineering → MI, RFE, polynomial",
            "5️⃣ Model Training → 7 algorithms with CV",
            "6️⃣ HP Optimization → Grid/Random/Bayesian",
            "7️⃣ Evaluation → Accuracy, F1, ROC-AUC",
            "8️⃣ Visualization → Plots & heatmaps",
            "9️⃣ Explainability → SHAP + LIME",
            "🔟 Research Paper → Auto-generated report",
        ]
        for step in steps:
            st.markdown(f"<div class='rec-box'>{step}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📋 CSV Format Requirements")
        st.markdown("""
        - Standard CSV format with header row
        - Target column can be anywhere in the file
        - Supports mixed data types (numeric + categorical)
        - Missing values handled automatically
        - Works with **any size**: n<30 to n>10,000

        #### 📦 Example Datasets Available
        - 🌸 **Iris** — Multiclass, clean data
        - 🎗️ **Breast Cancer** — Binary, real clinical data
        - 🍷 **Wine** — Multiclass, chemical features
        - 🔬 **Small Synthetic** — LOO-CV demo (n=60)
        - 🌙 **Moons** — Non-linear boundaries
        """)

else:
    # ── Data Preview Section ────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header"><h3>📋 Dataset Overview</h3></div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Target", target_col or "—")
    col4.metric("Missing %", f"{df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%")

    with st.expander("👀 Preview Data", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame(df.dtypes, columns=['Type']).reset_index()
            dtype_df.columns = ['Column', 'Type']
            dtype_df['Missing'] = df.isnull().sum().values
            dtype_df['Missing %'] = (df.isnull().sum().values / len(df) * 100).round(1)
            st.dataframe(dtype_df, use_container_width=True, height=200)
        with col2:
            st.write("**Statistics:**")
            st.dataframe(df.describe().round(3), use_container_width=True, height=200)

    # ── Run Pipeline Button ─────────────────────────────────────────────────
    st.markdown("---")
    if target_col:
        run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
        with run_col2:
            run_clicked = st.button(
                "🚀 Run Complete ML Pipeline",
                disabled=st.session_state.running,
            )
    else:
        st.warning("⚠️ Please select a target column in the sidebar.")
        run_clicked = False

    # ── Execute Pipeline ─────────────────────────────────────────────────────
    if run_clicked and target_col:
        st.session_state.running = True
        st.session_state.pipeline_results = None

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(msg: str, pct: int):
            progress_bar.progress(pct / 100)
            status_text.markdown(f"**{msg}**")

        pipeline = MLPipeline(
            output_dir="/tmp/ml_toolbox_outputs",
            progress_callback=update_progress,
        )

        with st.spinner(""):
            try:
                results = pipeline.run(
                    df=df,
                    target_col=target_col,
                    dataset_name=dataset_name,
                    task_type=task_type,
                    top_k_features=top_k,
                    hp_method=hp_method,
                    hp_n_iter=hp_n_iter,
                    run_explainability=run_explainability,
                    run_hp_optimization=run_hp,
                )
                st.session_state.pipeline_results = results
                st.session_state.pipeline = pipeline

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())

        st.session_state.running = False
        progress_bar.progress(1.0)
        status_text.success("✅ Pipeline completed successfully!")
        st.rerun()

    # ── Results Display ──────────────────────────────────────────────────────
    if st.session_state.pipeline_results:
        results = st.session_state.pipeline_results
        pipeline = st.session_state.pipeline

        summary = results.get("pipeline_summary", {})
        eval_report = results.get("evaluation", {})
        model_results = eval_report.get("model_results", {})
        best_model = eval_report.get("best_model", "N/A")
        best_result = model_results.get(best_model, {})

        st.markdown("---")
        st.markdown("""
        <div class="section-header"><h3>🏆 Pipeline Results</h3></div>
        """, unsafe_allow_html=True)

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best Model</div>
                <div class="metric-value" style="font-size:1.2rem;">{best_model}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            acc = best_result.get('accuracy', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{acc if acc == 'N/A' else f'{acc:.1%}'}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            f1 = best_result.get('f1_score', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{f1 if f1 == 'N/A' else f'{f1:.3f}'}</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            auc_val = best_result.get('roc_auc', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ROC-AUC</div>
                <div class="metric-value">{auc_val if auc_val == 'N/A' else f'{auc_val:.3f}'}</div>
            </div>""", unsafe_allow_html=True)
        with col5:
            elapsed = summary.get('elapsed_seconds', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pipeline Time</div>
                <div class="metric-value" style="font-size:1.4rem;">{elapsed}s</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs for different sections ─────────────────────────────────
        tabs = st.tabs([
            "🔍 Data Analysis",
            "📊 Model Results",
            "📈 Visualizations",
            "🧠 Explainability",
            "📝 Research Paper",
            "📦 Raw Reports",
        ])

        # ──── Tab 1: Data Analysis ──────────────────────────────────────
        with tabs[0]:
            data_report = results.get("data_understanding", {})
            prep_report = results.get("preprocessing", {})
            feat_report = results.get("feature_engineering", {})
            small_report = results.get("small_dataset", {})

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📋 Dataset Summary")
                summary_data = data_report.get("dataset_summary", {})
                for k, v in summary_data.items():
                    if k not in ['numeric_columns', 'categorical_columns']:
                        st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")

                st.markdown("#### 🔴 Missing Values")
                missing = data_report.get("missing_values", {})
                st.markdown(f"Total missing: **{missing.get('total_missing_pct', 0)}%**")
                if missing.get("columns_with_missing"):
                    mv_df = pd.DataFrame(missing["columns_with_missing"]).T
                    st.dataframe(mv_df, use_container_width=True)
                else:
                    st.success("✅ No missing values detected!")

                st.markdown("#### ⚠️ Outlier Detection")
                outliers = data_report.get("outliers", {})
                cols_with_outliers = outliers.get("columns_with_outliers", {})
                if cols_with_outliers:
                    out_df = pd.DataFrame(cols_with_outliers).T
                    st.dataframe(out_df[['iqr_outliers', 'iqr_outliers_pct', 'zscore_outliers']], use_container_width=True)
                else:
                    st.success("✅ No significant outliers detected!")

            with col2:
                st.markdown("#### 🎯 Class Distribution")
                class_dist = data_report.get("class_distribution", {})
                if class_dist:
                    dist_dict = class_dist.get("class_percentages", {})
                    if dist_dict:
                        dist_df = pd.DataFrame(list(dist_dict.items()), columns=['Class', 'Percentage'])
                        st.bar_chart(dist_df.set_index('Class'))
                        if class_dist.get("is_imbalanced"):
                            st.warning(f"⚖️ Imbalanced! Ratio: {class_dist.get('imbalance_ratio')}")

                st.markdown("#### ✅ Recommendations")
                recs = data_report.get("recommendations", [])
                for rec in recs:
                    st.markdown(f"<div class='rec-box'>{rec}</div>", unsafe_allow_html=True)

                st.markdown("#### 🔧 Preprocessing Steps Applied")
                steps = prep_report.get("steps_applied", [])
                for i, step in enumerate(steps):
                    st.markdown(f"<div class='rec-box'>{i+1}. {step}</div>", unsafe_allow_html=True)

            # Feature engineering results
            st.markdown("#### ⚙️ Feature Engineering — Mutual Information Scores")
            mi_scores = feat_report.get("mutual_information", {})
            if mi_scores and "error" not in mi_scores:
                valid_mi = {k: v for k, v in mi_scores.items() if isinstance(v, float)}
                if valid_mi:
                    mi_df = pd.DataFrame(
                        sorted(valid_mi.items(), key=lambda x: x[1], reverse=True),
                        columns=['Feature', 'MI Score']
                    ).set_index('Feature')
                    st.bar_chart(mi_df)

            # Small dataset optimization
            st.markdown("#### 📦 Small Dataset Optimization")
            cv_strategy = small_report.get("cv_strategy", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("CV Strategy", cv_strategy.get("strategy", "N/A"))
            col2.metric("CV Score", f"{cv_strategy.get('mean_score', 0):.3f} ± {cv_strategy.get('std_score', 0):.3f}")
            col3.metric("N Folds", cv_strategy.get("n_folds", "N/A"))

            if small_report.get("bootstrap_analysis"):
                ba = small_report["bootstrap_analysis"]
                st.info(f"🎲 Bootstrap Analysis (n=50): {ba.get('interpretation', 'N/A')}")

        # ──── Tab 2: Model Results ──────────────────────────────────────
        with tabs[1]:
            if pipeline:
                comparison_df = pipeline.get_model_comparison_df()
                if not comparison_df.empty:
                    st.markdown("#### 📊 Model Performance Comparison")

                    # Color-coded table
                    numeric_cols = comparison_df.select_dtypes(include=[float]).columns
                    st.dataframe(
                        comparison_df.set_index('Model').style.background_gradient(
                            cmap='RdYlGn', subset=[c for c in numeric_cols if 'accuracy' in c or 'f1' in c or 'roc' in c or 'r2' in c]
                        ).format("{:.4f}", subset=numeric_cols),
                        use_container_width=True, height=300
                    )

            # HP Optimization results
            hp_report = results.get("hyperparameter", {})
            if hp_report and hp_report.get("status") != "skipped":
                st.markdown("#### 🔬 Hyperparameter Optimization Results")
                for model_name, hp_result in hp_report.get("results", {}).items():
                    if "error" not in hp_result and "status" not in hp_result:
                        with st.expander(f"🔧 {model_name} — Best: {hp_result.get('best_score', 'N/A')}"):
                            col1, col2 = st.columns(2)
                            col1.metric("Method", hp_result.get("method", "N/A"))
                            col2.metric("Best Score", hp_result.get("best_score", "N/A"))
                            st.json(hp_result.get("best_params", {}))

            # Per-model detailed metrics
            st.markdown("#### 🔍 Detailed Per-Model Metrics")
            ranking = eval_report.get("ranking", [])
            for rank_item in ranking:
                model_name = rank_item["model"]
                result = model_results.get(model_name, {})
                if "error" not in result:
                    with st.expander(f"{'🥇' if model_name == best_model else '📊'} {model_name} — Score: {rank_item['score']:.4f}"):
                        cols = st.columns(5)
                        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
                        labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
                        for col, metric, label in zip(cols, metrics, labels):
                            val = result.get(metric)
                            col.metric(label, f"{val:.4f}" if val else "N/A")

                        # Confusion matrix
                        if "confusion_matrix" in result:
                            cm = result["confusion_matrix"]
                            cm_df = pd.DataFrame(cm)
                            st.markdown("**Confusion Matrix:**")
                            st.dataframe(cm_df, use_container_width=True)

        # ──── Tab 3: Visualizations ─────────────────────────────────────
        with tabs[2]:
            if pipeline:
                viz_agent = pipeline.get_visualization_agent()
                all_figures = viz_agent.get_all_figures()

                if all_figures:
                    # Model comparison
                    if "model_comparison" in all_figures:
                        st.markdown("#### 📊 Model Performance Comparison")
                        st.pyplot(all_figures["model_comparison"], use_container_width=True)

                    # ROC curves
                    if "roc_curves" in all_figures:
                        st.markdown("#### 📈 ROC Curves")
                        st.pyplot(all_figures["roc_curves"], use_container_width=True)

                    # Correlation heatmap
                    if "correlation_heatmap" in all_figures:
                        st.markdown("#### 🔗 Feature Correlation Heatmap")
                        st.pyplot(all_figures["correlation_heatmap"], use_container_width=True)

                    # Feature importance
                    if "feature_importance" in all_figures:
                        st.markdown("#### ⚙️ Feature Importance")
                        st.pyplot(all_figures["feature_importance"], use_container_width=True)

                    # Data distribution
                    if "data_distribution" in all_figures:
                        st.markdown("#### 📦 Feature Distributions")
                        st.pyplot(all_figures["data_distribution"], use_container_width=True)

                    # Confusion matrices
                    cm_keys = [k for k in all_figures.keys() if k.startswith("cm_")]
                    if cm_keys:
                        st.markdown("#### 🎯 Confusion Matrices")
                        cols = st.columns(min(2, len(cm_keys)))
                        for i, key in enumerate(cm_keys[:4]):
                            with cols[i % 2]:
                                st.pyplot(all_figures[key], use_container_width=True)
                else:
                    st.info("No visualizations generated yet.")

        # ──── Tab 4: Explainability ─────────────────────────────────────
        with tabs[3]:
            expl_report = results.get("explainability", {})

            if expl_report.get("status") == "skipped":
                st.info("Explainability was disabled in settings.")
            elif expl_report.get("status") == "failed":
                st.error(f"Explainability failed: {expl_report.get('error')}")
            else:
                if pipeline:
                    expl_agent = pipeline.get_explainability_agent()
                    expl_figures = expl_agent.get_figures()

                    # SHAP figures
                    shap_keys = [k for k in expl_figures.keys() if 'shap' in k.lower()]
                    if shap_keys:
                        st.markdown("#### 🔮 SHAP Analysis")
                        for key in shap_keys[:4]:
                            fig = expl_figures[key]
                            if fig:
                                st.pyplot(fig, use_container_width=True)

                    # LIME figures
                    lime_keys = [k for k in expl_figures.keys() if 'lime' in k.lower()]
                    if lime_keys:
                        st.markdown("#### 🔍 LIME Analysis")
                        for key in lime_keys[:3]:
                            fig = expl_figures[key]
                            if fig:
                                st.pyplot(fig, use_container_width=True)

                    # SHAP importance table
                    analyses = expl_report.get("analyses", {})
                    for model_name, analysis in analyses.items():
                        shap_data = analysis.get("shap", {})
                        if shap_data.get("status") == "success":
                            st.markdown(f"#### 📊 SHAP Feature Rankings — {model_name}")
                            importance = shap_data.get("feature_importance", {})
                            if importance:
                                imp_df = pd.DataFrame(
                                    list(importance.items())[:15],
                                    columns=['Feature', 'Mean |SHAP|']
                                )
                                st.dataframe(imp_df, use_container_width=True)

                            lime_data = analysis.get("lime", {})
                            if lime_data.get("status") == "success":
                                st.markdown(f"#### 📋 LIME Explanations — {model_name}")
                                for exp in lime_data.get("explanations", [])[:2]:
                                    with st.expander(f"Sample #{exp['instance']+1} (True label: {exp['true_label']})"):
                                        exp_df = pd.DataFrame(
                                            exp['explanation'],
                                            columns=['Feature/Rule', 'LIME Weight']
                                        )
                                        st.dataframe(exp_df, use_container_width=True)

                if not expl_figures and not expl_report.get("analyses"):
                    st.warning("No explainability data available. Check if SHAP and LIME are installed.")

        # ──── Tab 5: Research Paper ─────────────────────────────────────
        with tabs[4]:
            paper_report = results.get("research_paper", {})

            if paper_report:
                st.markdown(f"#### 📄 {paper_report.get('title', 'Research Report')}")
                st.caption(f"Generated: {paper_report.get('generated_at')}")

                paper_tabs = st.tabs(["Abstract", "Methodology", "Exp. Setup", "Results", "Discussion", "Conclusion", "LaTeX Table"])

                with paper_tabs[0]:
                    st.markdown(paper_report.get("abstract", ""))
                with paper_tabs[1]:
                    st.markdown(paper_report.get("methodology", ""))
                with paper_tabs[2]:
                    st.markdown(paper_report.get("experimental_setup", ""))
                with paper_tabs[3]:
                    st.markdown(paper_report.get("results", ""))
                with paper_tabs[4]:
                    st.markdown(paper_report.get("discussion", ""))
                with paper_tabs[5]:
                    st.markdown(paper_report.get("conclusion", ""))
                with paper_tabs[6]:
                    st.code(paper_report.get("latex_snippet", ""), language="latex")

                st.markdown("#### 📚 BibTeX References")
                for ref in paper_report.get("bibtex_references", []):
                    st.code(ref, language="bibtex")

                # Download full paper
                if pipeline:
                    full_text = pipeline.get_research_paper_text()
                    st.download_button(
                        "📥 Download Full Research Report (Markdown)",
                        data=full_text,
                        file_name=f"ml_research_report_{dataset_name.replace(' ', '_')}.md",
                        mime="text/markdown",
                    )

        # ──── Tab 6: Raw Reports ────────────────────────────────────────
        with tabs[5]:
            st.markdown("#### 🗃️ Raw Agent Reports (JSON)")

            report_keys = [
                "data_understanding", "preprocessing", "feature_engineering",
                "small_dataset", "model_training", "hyperparameter", "evaluation",
            ]

            for key in report_keys:
                if key in results:
                    with st.expander(f"📋 {key.replace('_', ' ').title()} Report"):
                        try:
                            # Clean for display
                            def clean_for_display(obj, depth=0):
                                if depth > 5:
                                    return str(obj)
                                if isinstance(obj, dict):
                                    return {k: clean_for_display(v, depth+1) for k, v in obj.items()
                                           if not callable(v) and k != 'best_estimator'}
                                if isinstance(obj, list):
                                    return [clean_for_display(i, depth+1) for i in obj[:20]]
                                if isinstance(obj, (np.integer, np.int64)):
                                    return int(obj)
                                if isinstance(obj, (np.floating, np.float64)):
                                    return float(obj)
                                if isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                return obj

                            clean_report = clean_for_display(results[key])
                            st.json(clean_report)
                        except Exception as e:
                            st.write(f"Error displaying report: {e}")

            # Download results JSON
            if pipeline:
                try:
                    json_path = pipeline.save_results()
                    with open(json_path, 'r') as f:
                        json_data = f.read()
                    st.download_button(
                        "📥 Download Full Results (JSON)",
                        data=json_data,
                        file_name=f"ml_pipeline_results_{dataset_name}.json",
                        mime="application/json",
                    )
                except Exception as e:
                    st.warning(f"Could not save JSON: {e}")


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4a5568; font-size: 0.8rem; padding: 1rem 0;">
    🤖 <strong>ML Toolbox</strong> — Modular AI Agent System |
    Built with scikit-learn, SHAP, LIME & Streamlit |
    Supports datasets from n&lt;30 to n&gt;10,000
</div>
""", unsafe_allow_html=True)
