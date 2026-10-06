import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import hashlib
from html import escape

from utils.helpers import (
    apply_custom_css,
    init_session_state,
    auto_detect_column_types,
    convert_df_to_csv,
    load_sample_dataset_cached,
    parse_uploaded_csv_cached,
    compute_data_summary_cached,
    compute_missing_table_cached,
    compute_outliers_table_cached,
    compute_descriptive_stats_cached,
    compute_text_similarity_cached
)

from modules.outlier import detect_outliers_iqr
from modules.feature_engineering import FeatureEngineer
from modules.preprocessing import run_pipeline_with_split
from modules.similarity import compute_single_pair_detail

# Apply CSS & Session State
apply_custom_css()
init_session_state()
if 'dataset_upload_digest' not in st.session_state:
    st.session_state.dataset_upload_digest = None
if 'dataset_uploader_version' not in st.session_state:
    st.session_state.dataset_uploader_version = 0
if st.session_state.pop('navigate_dataset_requested', False):
    st.session_state.main_navigation = "📁 Dataset"


def reset_dataset_dependent_state():
    """Clear results tied to the current tabular dataset without touching text inputs."""
    result_keys = (
        'pipeline_result', 'fe_dataset', 'fe_summary', 'processed_train',
        'processed_test', 'pipeline', 'preprocessing_result',
        'outlier_result', 'outlier_results', 'dataset_stats'
    )
    for key in result_keys:
        if key in st.session_state:
            st.session_state[key] = None
    for cached_function in (
        compute_data_summary_cached,
        compute_missing_table_cached,
        compute_outliers_table_cached,
        compute_descriptive_stats_cached
    ):
        cached_function.clear()
    st.session_state.dataset_upload_digest = None
    st.session_state.dataset_uploader_version += 1


def feature_config_for_dataset(df: pd.DataFrame) -> dict:
    """Enable only engineered features whose source columns exist in this dataset."""
    income_col = 'Income'
    expense_col = 'Monthly_Expense'
    age_col = 'Age'

    def has_numeric_values(column: str) -> bool:
        if column not in df.columns:
            return False
        return bool(pd.to_numeric(df[column], errors='coerce').notna().any())

    has_income = has_numeric_values(income_col)
    has_expense = has_numeric_values(expense_col)
    return {
        'enable_log_income': has_income,
        'enable_expense_ratio': has_income and has_expense,
        'enable_net_savings': has_income and has_expense,
        'enable_age_binning': has_numeric_values(age_col),
        'income_col': income_col,
        'expense_col': expense_col,
        'age_col': age_col
    }


def validate_active_dataset_for_pipeline(df: pd.DataFrame) -> list:
    """Return actionable issues before fitting the pipeline on an uploaded schema."""
    issues = []
    if len(df) < 5:
        issues.append("Minimal 5 baris data diperlukan untuk train/test split 80:20.")

    numerical_cols = st.session_state.numerical_cols
    nominal_cols = st.session_state.nominal_cols
    ordinal_cols = st.session_state.ordinal_cols
    selected_cols = numerical_cols + nominal_cols + ordinal_cols
    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Kolom konfigurasi tidak ditemukan: {', '.join(map(str, missing_cols))}.")

    if not selected_cols:
        issues.append("Pilih setidaknya satu kolom numerik, nominal, atau ordinal.")

    invalid_num_cols = [
        col for col in numerical_cols
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if invalid_num_cols:
        issues.append(f"Kolom numerik memiliki tipe non-numerik: {', '.join(map(str, invalid_num_cols))}.")

    empty_cols = [col for col in selected_cols if col in df.columns and df[col].isna().all()]
    if empty_cols:
        issues.append(f"Kolom berikut seluruh nilainya kosong: {', '.join(map(str, empty_cols))}.")

    missing_ordinal_config = [
        col for col in ordinal_cols
        if col in df.columns and col not in st.session_state.ordinal_categories
    ]
    if missing_ordinal_config:
        issues.append(f"Urutan kategori ordinal belum diatur untuk: {', '.join(map(str, missing_ordinal_config))}.")

    if not df.select_dtypes(include='number').columns.size and not (nominal_cols or ordinal_cols):
        issues.append("Dataset tidak memiliki kolom numerik atau kategorikal yang dapat diproses.")

    return issues


def execute_active_pipeline():
    """Fit the existing pipeline on the active dataset after validating its schema."""
    df = st.session_state.dataset
    if df is None:
        st.info("Belum ada dataset aktif. Silakan upload file CSV untuk memulai.")
        return None
    issues = validate_active_dataset_for_pipeline(df)
    if issues:
        st.warning("⚠️ Dataset belum dapat diproses")
        for issue in issues:
            st.write(f"- {issue}")
        st.info("Silakan periksa tipe atribut atau sesuaikan konfigurasi preprocessing.")
        return None

    try:
        result = run_pipeline_with_split(
            df=df,
            numerical_cols=st.session_state.numerical_cols,
            nominal_cols=st.session_state.nominal_cols,
            ordinal_cols=st.session_state.ordinal_cols,
            ordinal_categories_dict=st.session_state.ordinal_categories,
            fe_config=st.session_state.get('fe_config', {})
        )
    except (ValueError, TypeError, KeyError, IndexError) as error:
        st.warning("⚠️ Dataset belum dapat diproses")
        st.info("Silakan periksa tipe atribut, nilai kosong, dan konfigurasi preprocessing.")
        st.caption(f"Detail validasi: {error}")
        return None

    st.session_state.pipeline_result = result
    return result


# Helper to load sample dataset with cache
def load_default_sample():
    sample_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_dataset.csv')
    if not os.path.exists(sample_path):
        st.error("File data/sample_dataset.csv tidak ditemukan.")
        return False

    reset_dataset_dependent_state()
    load_sample_dataset_cached.clear(sample_path)
    df = load_sample_dataset_cached(sample_path)
    st.session_state.dataset = df
    st.session_state.dataset_name = "sample_dataset.csv (Bawaan)"
    st.session_state.dataset_upload_digest = None
    num_cols, nom_cols, ord_cols, ord_cats = auto_detect_column_types(df)
    st.session_state.numerical_cols = num_cols
    st.session_state.nominal_cols = nom_cols
    st.session_state.ordinal_cols = ord_cols
    st.session_state.ordinal_categories = ord_cats
    st.session_state.fe_config = feature_config_for_dataset(df)
    return True


def clear_active_dataset():
    """Return the app to its empty state without affecting text similarity inputs."""
    reset_dataset_dependent_state()
    st.session_state.dataset = None
    st.session_state.dataset_name = "Belum Ada Dataset"
    st.session_state.dataset_upload_digest = None
    st.session_state.numerical_cols = []
    st.session_state.nominal_cols = []
    st.session_state.ordinal_cols = []
    st.session_state.ordinal_categories = {}
    st.session_state.fe_config = feature_config_for_dataset(pd.DataFrame())


def activate_uploaded_dataset(file_name: str, file_bytes: bytes) -> bool:
    """Load and activate a CSV, clearing results from the previously active dataset."""
    upload_digest = hashlib.sha256(file_bytes).hexdigest()
    upload_name = f"{file_name} (Upload)"
    if (
        upload_digest == st.session_state.dataset_upload_digest
        and st.session_state.dataset_name == upload_name
    ):
        return False

    try:
        df_new = parse_uploaded_csv_cached(file_bytes)
        if df_new.empty or len(df_new.columns) == 0:
            st.error("CSV tidak berisi baris data dan kolom yang dapat digunakan.")
            return False

        reset_dataset_dependent_state()
        st.session_state.dataset = df_new
        st.session_state.dataset_name = upload_name
        st.session_state.dataset_upload_digest = upload_digest
        num_cols, nom_cols, ord_cols, ord_cats = auto_detect_column_types(df_new)
        st.session_state.numerical_cols = num_cols
        st.session_state.nominal_cols = nom_cols
        st.session_state.ordinal_cols = ord_cols
        st.session_state.ordinal_categories = ord_cats
        st.session_state.fe_config = feature_config_for_dataset(df_new)
        return True
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"CSV tidak dapat dibaca: {error}")
        return False

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
def navigate_to_dataset():
    st.session_state.navigate_dataset_requested = True


with st.sidebar:
    st.markdown("""
    <div class="brand-lockup">
        <div class="brand-name">DataPreprocess</div>
        <div class="brand-subtitle">Automated Data Preprocessing</div>
    </div>
                """, unsafe_allow_html=True)

    menu_options = {
        "🏠 Beranda": ("🏠 Beranda", "Ringkasan dataset aktif dan pipeline"),
        "📁 Dataset": ("📁 Dataset", "Unggah, tinjau, dan atur atribut"),
        "⚙️ Prapemrosesan": ("⚙️ Preprocessing", "Siapkan atribut untuk analisis"),
        "🛠️ Rekayasa Fitur": ("🛠️ Feature Engineering", "Buat atribut turunan yang berguna"),
        "🛡️ Pipeline & Data Leakage": ("🛡️ Pipeline & Data Leakage", "Tinjau pemrosesan yang aman dari kebocoran"),
        "📊 Hasil": ("📊 Hasil Preprocessing", "Tinjau data training dan testing"),
        "🔤 Cosine Similarity": ("🔤 Cosine Similarity", "Bandingkan teks dengan vektor biner"),
        "ℹ️ Tentang Sistem": ("ℹ️ Tentang Sistem", "Metode dan teknologi project")
    }

    selected_navigation = st.radio(
        "NAVIGASI",
        options=list(menu_options.keys()),
        key="main_navigation",
        label_visibility="collapsed"
    )
    menu_choice, menu_description = menu_options[selected_navigation]

    st.markdown(f"""
    <div class="nav-hint">
        <span class="nav-caption">NAVIGASI</span><br>{escape(menu_description)}
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.dataset is not None:
        n_rows, n_cols, n_missing = compute_data_summary_cached(st.session_state.dataset)
        dataset_status_html = f"""
        <div class="dataset-status">
            <div class="dataset-status-title">STATUS DATASET</div>
            <span class="status-pill active">● Dataset Aktif</span>
            <div class="dataset-file">{escape(st.session_state.dataset_name)}</div>
            <div class="dataset-meta">{n_rows:,} baris · {n_cols} kolom<br>{n_missing:,} missing values</div>
        </div>
        """
        if st.button("🧪 Gunakan Dataset Contoh", use_container_width=True):
            if load_default_sample():
                st.rerun()
        if st.button("🗑️ Hapus Dataset Aktif", use_container_width=True):
            clear_active_dataset()
            st.rerun()
    else:
        dataset_status_html = """
        <div class="dataset-status">
            <div class="dataset-status-title">STATUS DATASET</div>
            <span class="status-pill empty">● Belum Ada Dataset</span>
            <div class="dataset-meta" style="margin-top:.65rem">Unggah CSV untuk memulai</div>
        </div>
        """
        if st.button("📥 Unggah CSV", use_container_width=True):
            navigate_to_dataset()
            st.rerun()
        if st.button("🧪 Gunakan Dataset Contoh", use_container_width=True):
            if load_default_sample():
                st.rerun()
    st.markdown(dataset_status_html, unsafe_allow_html=True)

# Compact workspace marker; individual pages own their main heading.
# ==========================================
# 1. PAGE: BERANDA
# ==========================================
if menu_choice == "🏠 Beranda":
    df = st.session_state.dataset

    st.markdown("""
    <section class="hero">
        <div class="hero-copy">
            <div class="eyebrow">RUANG KERJA ANALITIK DATA</div>
            <h1>Automated Data Preprocessing</h1>
            <p>Siap Menganalisis Dataset? Unggah dataset CSV untuk melakukan prapemrosesan, rekayasa fitur, dan validasi data leakage secara otomatis.</p>
        </div>
        <div class="hero-mark">◈</div>
    </section>
    """, unsafe_allow_html=True)

    if df is None:
        st.markdown("""
        <div class="empty-state">
            <div>
                <div class="eyebrow">MULAI ANALISIS</div>
                <h2>📂 Mulai dengan Dataset</h2>
                <p>Unggah file CSV untuk memulai analisis dataset.</p>
            </div>
            <div class="empty-art">▤</div>
        </div>
        """, unsafe_allow_html=True)
        st.button("📥 Unggah Dataset", type="primary", use_container_width=True, on_click=navigate_to_dataset)
        if st.button("🧪 Gunakan Dataset Contoh", use_container_width=True):
            if load_default_sample():
                st.rerun()
        st.stop()

    n_rows, n_cols, n_missing = compute_data_summary_cached(df)
    numeric_cols = tuple(col for col in st.session_state.numerical_cols if col in df.columns)
    outlier_count = 0
    if numeric_cols:
        outlier_table = compute_outliers_table_cached(df, numeric_cols)
        outlier_count = int(outlier_table['Jumlah Outlier'].sum())
    processed_feature_count = (
        len(st.session_state.pipeline_result['feature_names'])
        if st.session_state.pipeline_result is not None else 0
    )

    metric_columns = st.columns(4)
    metric_data = [
        ("TOTAL DATA", f"{n_rows:,}", "baris dalam dataset aktif"),
        ("MISSING VALUES", f"{n_missing:,}", "nilai yang perlu diimputasi"),
        ("OUTLIER", f"{outlier_count:,}", "kandidat metode IQR"),
        ("FITUR HASIL", f"{processed_feature_count:,}", "jalankan pipeline untuk memperbarui")
    ]
    for column, (label, value, note) in zip(metric_columns, metric_data):
        with column:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-heading'><span class='eyebrow'>RINGKASAN PIPELINE</span><h2>Dari data mentah menjadi fitur siap model</h2></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="workflow-grid">
        <div class="workflow-card"><div class="workflow-icon">↔</div><div class="workflow-title">Train / Test Split</div><div class="workflow-desc">Data dibagi terlebih dahulu. Data testing tidak digunakan saat fit.</div></div>
        <div class="workflow-card"><div class="workflow-icon">✦</div><div class="workflow-title">Feature Engineering</div><div class="workflow-desc">Atribut turunan dibuat di dalam Scikit-Learn Pipeline.</div></div>
        <div class="workflow-card"><div class="workflow-icon">✓</div><div class="workflow-title">Fit pada Data Training</div><div class="workflow-desc">Imputer, batas, scaler, dan encoder dipelajari dari training saja.</div></div>
        <div class="workflow-card"><div class="workflow-icon">⇢</div><div class="workflow-title">Transform Data Testing</div><div class="workflow-desc">Terapkan parameter hasil training tanpa fit ulang pada data testing.</div></div>
    </div>
    """, unsafe_allow_html=True)

    action_col, status_col = st.columns([1.05, 1.95])
    with action_col:
        if st.button("▶ Jalankan Pipeline Prapemrosesan", type="primary", use_container_width=True):
            result = execute_active_pipeline()
            if result is not None:
                st.rerun()
    with status_col:
        if st.session_state.pipeline_result is None:
            st.info("Pipeline belum dijalankan untuk dataset aktif.")
        else:
            st.success(f"Pipeline siap · {len(st.session_state.pipeline_result['feature_names'])} fitur hasil")

    st.stop()
    
    # 1. Purpose Card
    st.markdown("""
    <div class="purpose-card">
        <div class="purpose-title">🎯 Tujuan Sistem</div>
        <div class="purpose-text">
            Sistem ini mengolah dataset mentah yang memiliki missing values, outliers, serta atribut numerik, nominal, dan ordinal menjadi data yang siap digunakan untuk Machine Learning secara otomatis dan bebas dari kebocoran data (data leakage).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Key Metrics Row
    if df is not None:
        n_rows, n_cols, n_missing = compute_data_summary_cached(df)
        num_cols_tuple = tuple(st.session_state.numerical_cols)
        
        # Outlier calculation with cache
        if num_cols_tuple:
            out_df = compute_outliers_table_cached(df, num_cols_tuple)
            n_outliers = int(out_df['Jumlah Outlier'].sum())
        else:
            n_outliers = 0

        n_proc = len(st.session_state.pipeline_result['feature_names']) if st.session_state.pipeline_result is not None else "-"

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📊 Total Data</div>
                <div class="metric-value">{n_rows:,}</div>
                <div class="metric-sub">Baris data mentah</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🔢 Jumlah Atribut</div>
                <div class="metric-value">{n_cols}</div>
                <div class="metric-sub">Kolom input awal</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">⚠️ Missing Values</div>
                <div class="metric-value" style="color: {'#EF4444' if n_missing > 0 else '#10B981'};">{n_missing}</div>
                <div class="metric-sub">Nilai kosong terdeteksi</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📦 Outliers</div>
                <div class="metric-value" style="color: #F59E0B;">{n_outliers}</div>
                <div class="metric-sub">Pencilan metode IQR</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">✨ Fitur Pipeline</div>
                <div class="metric-value" style="color: #2563EB;">{n_proc}</div>
                <div class="metric-sub">Fitur terproses siap model</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Main Action Button
        col_btn_center, col_btn_info = st.columns([1.2, 1])
        with col_btn_center:
            st.markdown("""
            <div class="section-card">
                <div class="section-title">🚀 Eksekusi Pipeline Utama</div>
                <p style="font-size: 13px; color: #475569; margin-bottom: 14px;">
                    Jalankan seluruh rangkaian Feature Engineering, Imputasi, Outlier Capping, Scaling, dan Encoding dalam satu Scikit-Learn Pipeline.
                </p>
            """, unsafe_allow_html=True)

            if st.button("🚀 Jalankan Preprocessing Pipeline Lengkap", type="primary", use_container_width=True):
                res = execute_active_pipeline()
                if res is not None:
                    st.success("✅ **Pipeline berhasil dijalankan!**")
                    st.markdown(f"""
                    <div style='background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 10px; margin-top: 6px; font-size: 12.5px; color: #166534;'>
                        ✓ <strong>{len(res['X_train_proc']):,} data training</strong> berhasil diproses.<br>
                        ✓ <strong>{len(res['X_test_proc']):,} data testing</strong> berhasil ditransformasi tanpa data leakage.<br>
                        ✓ Total <strong>{len(res['feature_names'])} fitur akhir</strong> siap digunakan.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        with col_btn_info:
            st.markdown("""
            <div class="leakage-shield-box">
                <div class="leakage-title">🔐 Pencegahan Data Leakage</div>
                <div class="leakage-desc">
                    Data dibagi menjadi <strong>training</strong> dan <strong>testing</strong> terlebih dahulu. Parameter preprocessing (median imputasi, IQR bounds, mean & std scaling, kategori encoding) dipelajari <strong>hanya dari data training</strong>. Data testing hanya ditransformasi menggunakan parameter tersebut.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 4. System Flow Section
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🔄 Alur Sistem Preprocessing</div>
            <div style="font-size: 13px; color: #475569; margin-bottom: 12px;">
                Tahapan prapemrosesan data berjalan berurutan secara terotomasi di dalam Scikit-Learn Pipeline:
            </div>
            <div class="flow-wrapper">
                <div class="flow-node active">1. DATASET MENTAH</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-node">2. TRAIN / TEST SPLIT</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-node">3. MISSING VALUE</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-node">4. OUTLIER HANDLING</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-node">5. FEATURE ENGINEERING</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-node">6. SCALING & ENCODING</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-node active">7. DATA TERPROSES</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5. Academic Concept Section
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📚 Apa yang Dilakukan Sistem? (Konsep Akademik)</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px;">
                <div class="concept-item">
                    <div class="concept-title">1. Imputasi</div>
                    <div class="concept-desc">Mengisi data yang kosong menggunakan median (numerik) dan modus (kategorikal).</div>
                </div>
                <div class="concept-item">
                    <div class="concept-title">2. Outlier Handling</div>
                    <div class="concept-desc">Menangani nilai ekstrem menggunakan metode IQR Capping agar model tidak terdistorsi.</div>
                </div>
                <div class="concept-item">
                    <div class="concept-title">3. Scaling</div>
                    <div class="concept-desc">Menyamakan skala atribut numerik dengan StandardScaler (mean=0, std=1).</div>
                </div>
                <div class="concept-item">
                    <div class="concept-title">4. One-Hot Encoding</div>
                    <div class="concept-desc">Mengubah kategori nominal tanpa urutan (misal: Gender, City) menjadi fitur biner.</div>
                </div>
                <div class="concept-item">
                    <div class="concept-title">5. Ordinal Encoding</div>
                    <div class="concept-desc">Mengubah kategori bertingkat (misal: Satisfaction) menjadi urutan numerik (0, 1, 2...).</div>
                </div>
                <div class="concept-item">
                    <div class="concept-title">6. Feature Engineering</div>
                    <div class="concept-desc">Membuat fitur baru yang relevan (Log_Income, Rasio, Net_Savings, Age_Group).</div>
                </div>
                <div class="concept-item" style="border-left-color: #10B981;">
                    <div class="concept-title">7. Data Leakage Prevention</div>
                    <div class="concept-desc">Memastikan data testing tidak pernah digunakan untuk mempelajari parameter preprocessing.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Belum ada dataset aktif. Silakan upload file CSV untuk memulai.")


# ==========================================
# 2. PAGE: DATASET
# ==========================================
elif menu_choice == "📁 Dataset":
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">MANAJEMEN DATA</div>
        <h1>Manajemen Dataset</h1>
        <p>Tinjau sumber data aktif dan atur tipe atribut sebelum prapemrosesan.</p>
    </div>
    """, unsafe_allow_html=True)
    df = st.session_state.dataset
    with st.container(border=True):
        st.markdown("#### Dataset Aktif")
        if df is None:
            st.write("Belum ada dataset")
        else:
            st.markdown(f"**{st.session_state.dataset_name}**")
            active_rows, active_cols, active_missing = compute_data_summary_cached(df)
            active_metric_1, active_metric_2, active_metric_3 = st.columns(3)
            active_metric_1.metric("Baris", f"{active_rows:,}")
            active_metric_2.metric("Kolom", active_cols)
            active_metric_3.metric("Missing Values", active_missing)
        st.caption("Dataset aktif digunakan oleh seluruh tahapan Pipeline prapemrosesan.")

    with st.container(border=True):
        st.markdown("#### Unggah Dataset CSV")
        st.write("Unggah dataset CSV untuk diproses menggunakan Pipeline prapemrosesan.")
        uploaded = st.file_uploader(
            "📂 Unggah Dataset CSV",
            type=["csv"],
            key=f"dataset_upload_{st.session_state.dataset_uploader_version}"
        )
        if uploaded is not None:
            if activate_uploaded_dataset(uploaded.name, uploaded.getvalue()):
                st.success(f"Dataset {uploaded.name} berhasil menjadi dataset aktif.")
                st.rerun()

        if df is None:
            if st.button("🧪 Gunakan Dataset Contoh", use_container_width=True):
                if load_default_sample():
                    st.rerun()
        else:
            action_sample, action_clear = st.columns(2)
            with action_sample:
                if st.button("🧪 Gunakan Dataset Contoh", use_container_width=True):
                    if load_default_sample():
                        st.rerun()
            with action_clear:
                if st.button("🗑️ Hapus Dataset Aktif", use_container_width=True):
                    clear_active_dataset()
                    st.rerun()

    df = st.session_state.dataset
    if df is None:
        st.info("Belum ada dataset aktif. Unggah file CSV atau pilih dataset contoh untuk memulai.")
    else:
        st.markdown("---")
        
        # Tabs for Clean Dataset Inspection
        tab_prev, tab_type, tab_miss, tab_stat = st.tabs([
            "Ringkasan",
            "Tipe Atribut",
            "Missing Values",
            "Statistik"
        ])

        with tab_prev:
            st.markdown("#### Pratinjau Data")
            st.caption(f"Menampilkan {min(10, len(df))} baris pertama · {len(df.columns)} atribut")
            st.dataframe(df.head(10), use_container_width=True, height=360)

        with tab_type:
            st.markdown("#### Tipe Atribut")
            type_count_columns = st.columns(3)
            type_count_columns[0].metric("Numerik", len(st.session_state.numerical_cols))
            type_count_columns[1].metric("Nominal", len(st.session_state.nominal_cols))
            type_count_columns[2].metric("Ordinal", len(st.session_state.ordinal_cols))
            st.caption("Tinjau tipe yang terdeteksi atau sesuaikan sebelum menjalankan Pipeline.")
            
            all_cols = list(df.columns)
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                sel_num = st.multiselect("🔢 Numerik", all_cols, default=[c for c in st.session_state.numerical_cols if c in all_cols])
            with col_t2:
                rem_nom = [c for c in all_cols if c not in sel_num]
                sel_nom = st.multiselect("🏷️ Nominal", rem_nom, default=[c for c in st.session_state.nominal_cols if c in rem_nom])
            with col_t3:
                rem_ord = [c for c in all_cols if c not in sel_num and c not in sel_nom]
                sel_ord = st.multiselect("📊 Ordinal", rem_ord, default=[c for c in st.session_state.ordinal_cols if c in rem_ord])

            if sel_ord:
                st.markdown("##### Urutan Ordinal")
                ord_dict_up = {}
                for o_col in sel_ord:
                    u_vals = list(df[o_col].dropna().unique())
                    def_ord = st.session_state.ordinal_categories.get(o_col, u_vals)
                    ord_str = st.text_input(f"Urutan {o_col} (terendah ke tertinggi, pisahkan dengan koma):", value=", ".join([str(x) for x in def_ord]))
                    ord_dict_up[o_col] = [x.strip() for x in ord_str.split(",") if x.strip()]

            if st.button("💾 Simpan Tipe Atribut", type="primary"):
                st.session_state.numerical_cols = sel_num
                st.session_state.nominal_cols = sel_nom
                st.session_state.ordinal_cols = sel_ord
                if sel_ord:
                    st.session_state.ordinal_categories = ord_dict_up
                st.session_state.pipeline_result = None
                st.success("✓ Pengaturan tipe kolom berhasil disimpan!")
                st.rerun()

        with tab_miss:
            st.markdown("#### Missing Value Profile")
            miss_df = compute_missing_table_cached(df)
            st.dataframe(miss_df[['Atribut', 'Missing', 'Persentase']], use_container_width=True, height=360)

        with tab_stat:
            st.markdown("#### Statistik Numerik")
            if st.session_state.numerical_cols:
                stats_df = compute_descriptive_stats_cached(df, tuple(st.session_state.numerical_cols))
                st.dataframe(stats_df, use_container_width=True, height=360)
            else:
                st.info("Belum ada atribut numerik yang dipilih.")


# ==========================================
# 3. PAGE: PREPROCESSING
# ==========================================
elif menu_choice == "⚙️ Preprocessing":
    df = st.session_state.dataset
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">KONFIGURASI PIPELINE</div>
        <h1>Prapemrosesan</h1>
        <p>Tinjau transformasi dalam Scikit-Learn Pipeline yang aman dari Data Leakage.</p>
    </div>
    """, unsafe_allow_html=True)
    if df is None:
        st.info("Belum ada dataset aktif. Silakan upload file CSV untuk memulai.")
    else:
        st.markdown("""
        <div class="workflow-grid">
            <div class="workflow-card"><div class="workflow-icon">∅</div><div class="workflow-title">Missing Values</div><div class="workflow-desc">Median untuk numerik, modus untuk kategorikal.</div><span class="badge badge-success">✓ Imputasi</span></div>
            <div class="workflow-card"><div class="workflow-icon">⌁</div><div class="workflow-title">Penanganan Outlier</div><div class="workflow-desc">Batas IQR dipelajari dari data training lalu diterapkan konsisten.</div><span class="badge badge-info">IQR Capping</span></div>
            <div class="workflow-card"><div class="workflow-icon">▦</div><div class="workflow-title">Encoding</div><div class="workflow-desc">One-Hot Encoding untuk nominal dan Ordinal Encoding untuk atribut bertingkat.</div><span class="badge badge-info">Pemetaan kategori</span></div>
            <div class="workflow-card"><div class="workflow-icon">↗</div><div class="workflow-title">Scaling</div><div class="workflow-desc">Skalakan fitur numerik menggunakan parameter data training.</div><span class="badge badge-success">✓ Fit pada training</span></div>
        </div>
        """, unsafe_allow_html=True)

        tab_m, tab_o, tab_s, tab_e = st.tabs([
            "Missing Values",
            "Penanganan Outlier",
            "Scaling",
            "Encoding"
        ])

        with tab_m:
            st.markdown("#### Imputasi Missing Values")
            st.markdown("""
            Missing Values dapat menghambat proses analisis. Pipeline menggunakan `SimpleImputer`:
                - **Numerik:** median, lebih tahan terhadap kemencengan distribusi dan pencilan.
            - **Kategorikal:** kategori yang paling sering muncul.
            """)
            miss_df = compute_missing_table_cached(df)
            st.dataframe(miss_df[['Atribut', 'Missing', 'Persentase']], use_container_width=True, height=330)

        with tab_o:
            st.markdown("#### Penanganan Outlier dengan IQR")
            st.markdown("""
            Outlier dibatasi menggunakan rentang Interquartile Range yang dipelajari dari data training.
            - $IQR = Q3 - Q1$
            - $\\text{Lower Bound} = Q1 - 1.5 \\times IQR$
            - $\\text{Upper Bound} = Q3 + 1.5 \\times IQR$
            """)
            if st.session_state.numerical_cols:
                out_df = compute_outliers_table_cached(df, tuple(st.session_state.numerical_cols))
                st.dataframe(out_df, use_container_width=True, height=330)
                
                # Interactive Boxplot
                col_box = st.selectbox("Pilih Kolom untuk Visualisasi Boxplot:", st.session_state.numerical_cols)
                if col_box:
                    fig, ax = plt.subplots(figsize=(7, 2.5))
                    sns.boxplot(x=df[col_box].dropna(), color='#93C5FD', ax=ax)
                    ax.set_title(f"Boxplot Distribusi: {col_box}", fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

        with tab_s:
            st.markdown("#### Scaling Fitur")
            st.markdown("""
            Scaling menyamakan rentang fitur numerik agar nilai berskala besar tidak mendominasi model.
            - Default: `StandardScaler` (normalisasi Z-score).
            - Parameter hanya di-fit pada data training.
            """)
            scaler_choice = st.selectbox("Pilih scaler", ["StandardScaler (Direkomendasikan)", "MinMaxScaler (Rentang 0-1)", "RobustScaler (Median & IQR)"])
            st.session_state.scaler_type_choice = 'standard' if 'Standard' in scaler_choice else ('minmax' if 'MinMax' in scaler_choice else 'robust')

        with tab_e:
            st.markdown("#### Encoding Kategorikal")
            st.markdown("""
            Atribut kategorikal diubah menjadi representasi numerik.
            - **Nominal:** `OneHotEncoder` membuat indikator biner.
            - **Ordinal:** `OrdinalEncoder` mengikuti urutan kategori yang dikonfigurasi.
            """)
            st.markdown("##### Contoh Encoding")
            sample_enc = pd.DataFrame({
                'Data Mentah (Gender)': ['Male', 'Female', 'Male'],
                'Data Mentah (Satisfaction)': ['Neutral', 'Very Unsatisfied', 'Very Satisfied'],
                'Hasil Encoding (Gender_Male)': [1, 0, 1],
                'Hasil Encoding (Gender_Female)': [0, 1, 0],
                'Hasil Encoding (Satisfaction)': [2, 0, 4]
            })
            st.dataframe(sample_enc, use_container_width=True)


# ==========================================
# 4. PAGE: FEATURE ENGINEERING
# ==========================================
elif menu_choice == "🛠️ Feature Engineering":
    df = st.session_state.dataset
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">REKAYASA FITUR</div>
        <h1>Rekayasa Fitur</h1>
        <p>Atribut turunan dibuat di dalam Scikit-Learn Pipeline dan dipelajari hanya dari data training.</p>
    </div>
    """, unsafe_allow_html=True)
    if df is None:
        st.info("Belum ada dataset aktif. Silakan upload file CSV untuk memulai.")
    else:
        current_fe = st.session_state.get('fe_config', {
            'enable_log_income': True,
            'enable_expense_ratio': True,
            'enable_net_savings': True,
            'enable_age_binning': True
        })

        fe_cards = [
            ("⌁", "Log Income", "Income", "Log_Income", "Transformasi logaritmik pada atribut Income."),
            ("↗", "Expense to Income Ratio", "Monthly_Expense + Income", "Expense_to_Income_Ratio", "Mengukur proporsi pengeluaran bulanan terhadap pendapatan."),
            ("＋", "Net Savings", "Income + Monthly_Expense", "Net_Savings", "Menghitung sisa pendapatan setelah pengeluaran bulanan."),
            ("▥", "Age Group", "Age", "Age_Group", "Mengelompokkan usia ke dalam kategori demografis.")
        ]
        feature_cards_html = "".join(
            f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-desc">{description}</div><div class="feature-formula">{source} → {output}</div></div>'
            for icon, title, source, output, description in fe_cards
        )
        st.markdown(f"<div class='feature-grid'>{feature_cards_html}</div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Before and After Feature Count
        fe_transformer = FeatureEngineer(
            enable_log_income=current_fe.get('enable_log_income', True),
            enable_expense_ratio=current_fe.get('enable_expense_ratio', True),
            enable_net_savings=current_fe.get('enable_net_savings', True),
            enable_age_binning=current_fe.get('enable_age_binning', True)
        )
        fe_transformer.fit(df)
        df_preview = fe_transformer.transform(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("FITUR ASAL", df.shape[1], help="Jumlah kolom dataset awal")
        c2.metric("FITUR SETELAH REKAYASA", df_preview.shape[1], delta=f"+{df_preview.shape[1] - df.shape[1]}")
        c3.metric("FITUR TURUNAN", len(fe_transformer.created_features_))

        st.markdown("##### Pratinjau data hasil rekayasa")
        preview_cols = list(dict.fromkeys(
            [col for col in ['Age', 'Income', 'Monthly_Expense'] if col in df_preview.columns]
            + [feature['Nama Fitur'] for feature in fe_transformer.created_features_]
        ))
        if preview_cols:
            st.dataframe(df_preview[preview_cols].head(6), use_container_width=True)
        else:
            st.dataframe(df_preview.head(6), use_container_width=True)


# ==========================================
# 5. PAGE: PIPELINE & DATA LEAKAGE
# ==========================================
elif menu_choice == "🛡️ Pipeline & Data Leakage":
    df = st.session_state.dataset
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">VALIDASI & KEPERCAYAAN</div>
        <h1>Pipeline & Data Leakage</h1>
        <p>Parameter prapemrosesan dipelajari dari data training; data testing hanya ditransformasi dengan parameter tersebut.</p>
    </div>
    """, unsafe_allow_html=True)
    if df is None:
        st.info("Belum ada dataset aktif. Silakan upload file CSV untuk memulai.")
    else:
        st.markdown("""
        <div class="leakage-grid">
            <div class="leakage-lane train-lane">
                <div class="lane-heading"><span class="lane-dot"></span> JALUR DATA TRAINING · 80%</div>
                <div class="lane-step">DATA MENTAH</div><div class="lane-arrow">↓</div>
                <div class="lane-step">TRAIN / TEST SPLIT</div><div class="lane-arrow">↓</div>
                <div class="lane-step emphasized">DATA TRAINING → FIT PREPROCESSING</div><div class="lane-arrow">↓</div>
                <div class="lane-step">TRANSFORM DATA TRAINING</div>
            </div>
            <div class="leakage-lane test-lane">
                <div class="lane-heading"><span class="lane-dot"></span> JALUR DATA TESTING · 20%</div>
                <div class="lane-step muted-step">DATA TESTING</div><div class="lane-arrow">↓</div>
                <div class="lane-step emphasized">TRANSFORM SAJA</div><div class="lane-arrow">↓</div>
                <div class="lane-step muted-step">DATA TESTING SIAP MODEL</div>
                <div class="no-refit">TIDAK ADA REFIT PADA DATA TESTING</div>
            </div>
        </div>
        <div class="leakage-shield-box">
            <div class="leakage-title">✓ AMAN DARI DATA LEAKAGE</div>
            <div class="leakage-desc">Parameter prapemrosesan dipelajari hanya dari data training. Data testing diproses dengan <code>pipeline.transform()</code> tanpa fit ulang.</div>
        </div>
        """, unsafe_allow_html=True)

        # Learned parameters display
        if st.session_state.pipeline_result is not None:
            st.markdown("#### Parameter yang Dipelajari dari Data Training")
            res = st.session_state.pipeline_result
            params = res.get('learned_params', {})
            with st.expander("Lihat parameter hasil fit"):
                for title, p_val in params.items():
                    st.markdown(f"**{title}**")
                    st.json(p_val)
        else:
            st.info("Jalankan Pipeline untuk melihat parameter yang dipelajari dari data training.")

        if st.button("▶ Jalankan Pipeline Prapemrosesan", type="primary", use_container_width=True):
            res = execute_active_pipeline()
            if res is not None:
                st.success("Pipeline selesai tanpa fit pada data testing.")
                st.rerun()


# ==========================================
# 6. PAGE: HASIL PREPROCESSING
# ==========================================
elif menu_choice == "📊 Hasil Preprocessing":
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">HASIL PIPELINE</div>
        <h1>Hasil</h1>
        <p>Bandingkan data mentah, tinjau hasil transformasi, dan periksa parameter yang dipelajari.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.dataset is None:
        st.info("Belum ada dataset aktif. Silakan upload file CSV untuk memulai.")
    elif st.session_state.pipeline_result is None:
        with st.container(border=True):
            st.markdown("#### Belum Ada Hasil")
            st.write("Jalankan Pipeline untuk dataset aktif agar hasil tampil di sini.")
        if st.button("▶ Jalankan Pipeline Prapemrosesan", type="primary"):
            res = execute_active_pipeline()
            if res is not None:
                st.rerun()
    else:
        res = st.session_state.pipeline_result
        raw_train = res['X_train_raw']
        proc_train = res['X_train_proc']
        proc_test = res['X_test_proc']

        metric_row = st.columns(4)
        metric_row[0].metric("BARIS TRAINING", f"{len(proc_train):,}")
        metric_row[1].metric("BARIS TESTING", f"{len(proc_test):,}")
        metric_row[2].metric("FITUR ASAL", raw_train.shape[1])
        metric_row[3].metric("FITUR HASIL", proc_train.shape[1])

        overview_tab, training_tab, testing_tab, transform_tab, statistics_tab = st.tabs([
            "Ringkasan", "Data Training", "Data Testing", "Transformasi", "Statistik"
        ])
        with overview_tab:
            st.markdown("#### Ringkasan pembagian data")
            overview_train, overview_test = st.columns(2)
            with overview_train:
                st.markdown("**Data training · pratinjau mentah**")
                st.dataframe(res['X_train_raw'].head(6), use_container_width=True, height=280)
            with overview_test:
                st.markdown("**Data testing · pratinjau mentah**")
                st.dataframe(res['X_test_raw'].head(6), use_container_width=True, height=280)
            st.success(f"Pembagian aman dari Data Leakage · {len(proc_train):,} baris training · {len(proc_test):,} baris testing")

        with training_tab:
            st.markdown("#### Data training hasil prapemrosesan")
            st.dataframe(proc_train.round(3), use_container_width=True, height=420)
            st.download_button("⬇ Unduh CSV Data Training", convert_df_to_csv(proc_train), "processed_train_dataset.csv", "text/csv", type="primary")

        with testing_tab:
            st.markdown("#### Data testing hasil prapemrosesan")
            st.dataframe(proc_test.round(3), use_container_width=True, height=420)
            st.download_button("⬇ Unduh CSV Data Testing", convert_df_to_csv(proc_test), "processed_test_dataset.csv", "text/csv")

        with transform_tab:
            st.markdown("#### Transformasi fitur")
            created_features = pd.DataFrame(res.get('created_features', []))
            if not created_features.empty:
                st.dataframe(created_features.rename(columns={
                    'Nama Fitur': 'Nama Fitur', 'Tipe': 'Tipe', 'Formula': 'Rumus', 'Tujuan / Alasan': 'Tujuan / Alasan'
                }), use_container_width=True, height=260)
            else:
                st.info("Tidak ada fitur turunan yang aktif untuk dataset ini.")
            with st.expander("Parameter prapemrosesan yang dipelajari"):
                for parameter_name, parameter_value in res.get('learned_params', {}).items():
                    st.markdown(f"**{parameter_name}**")
                    st.json(parameter_value)

        with statistics_tab:
            st.markdown("#### Ringkasan statistik")
            stats_raw, stats_processed = st.columns(2)
            with stats_raw:
                st.markdown("**Data training mentah**")
                st.dataframe(raw_train.describe().T.rename(columns={
                    'count': 'jumlah', 'mean': 'rata-rata', 'std': 'simpangan baku', 'min': 'minimum',
                    '25%': '25%', '50%': '50%', '75%': '75%', 'max': 'maksimum'
                }).round(2), use_container_width=True, height=360)
            with stats_processed:
                st.markdown("**Data training terproses**")
                st.dataframe(proc_train.describe().T.rename(columns={
                    'count': 'jumlah', 'mean': 'rata-rata', 'std': 'simpangan baku', 'min': 'minimum',
                    '25%': '25%', '50%': '50%', '75%': '75%', 'max': 'maksimum'
                }).round(3), use_container_width=True, height=360)


# ==========================================
# 7. PAGE: COSINE SIMILARITY
# ==========================================
elif menu_choice == "🔤 Cosine Similarity":
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">MODUL TAMBAHAN · ANALISIS TEKS</div>
        <h1>Cosine Similarity Teks</h1>
        <p>Mengukur tingkat kemiripan antar teks menggunakan representasi vektor biner.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Pilihan Metode Input (Radio / Tabs)
    st.markdown("#### Dokumen Teks")

    input_mode = st.radio(
        "Mode input",
        ["✍️ Input Manual", "📂 Upload File"],
        index=0 if st.session_state.text_input_mode == "manual" else 1,
        horizontal=True,
        key="similarity_input_mode",
    )

    if input_mode == "✍️ Input Manual":
        st.session_state.text_input_mode = "manual"
    else:
        st.session_state.text_input_mode = "file"

    active_texts = []

    # ----------------------------------------------------
    # METODE 1: INPUT MANUAL
    # ----------------------------------------------------
    if st.session_state.text_input_mode == "manual":
        st.caption("Input manual cocok untuk beberapa teks pendek.")

        col_b1, col_b2, col_b3 = st.columns([1, 1, 1.2])
        with col_b1:
            if st.button("➕ Tambah Teks", use_container_width=True):
                st.session_state.manual_texts.append("")
                st.session_state.manual_text_ids.append(st.session_state.manual_text_next_id)
                st.session_state.manual_text_next_id += 1
                st.rerun()
        with col_b2:
            if st.button("🗑️ Hapus Semua", use_container_width=True):
                st.session_state.manual_texts = []
                st.session_state.manual_text_ids = []
                st.rerun()
        with col_b3:
            if st.button("🔄 Reset Contoh", use_container_width=True):
                st.session_state.manual_texts = [
                    "Aku suka membaca buku",
                    "Aku suka membaca novel",
                    "Hari ini cuaca sangat cerah"
                ]
                st.session_state.manual_text_ids = [
                    st.session_state.manual_text_next_id + offset
                    for offset in range(len(st.session_state.manual_texts))
                ]
                st.session_state.manual_text_next_id += len(st.session_state.manual_texts)
                st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if len(st.session_state.manual_texts) > 0:
            del_idx = None
            for idx, item_id in enumerate(st.session_state.manual_text_ids):
                c_txt, c_btn = st.columns([6, 1])
                with c_txt:
                    val = st.text_input(
                        f"S{idx + 1}:",
                        value=st.session_state.manual_texts[idx],
                        key=f"man_inp_{item_id}",
                        placeholder=f"Masukkan dokumen S{idx + 1}..."
                    )
                    st.session_state.manual_texts[idx] = val
                with c_btn:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️ Hapus", key=f"btn_del_man_{item_id}", use_container_width=True):
                        del_idx = idx

            if del_idx is not None:
                st.session_state.manual_texts.pop(del_idx)
                st.session_state.manual_text_ids.pop(del_idx)
                st.rerun()

        active_texts = st.session_state.manual_texts[:]

    # ----------------------------------------------------
    # METODE 2: UPLOAD FILE (CSV / TXT)
    # ----------------------------------------------------
    else:
        st.caption("Unggah CSV atau TXT untuk kumpulan dokumen teks yang lebih besar.")
        
        file_uploader = st.file_uploader(
            "Unggah dokumen teks (.csv atau .txt)",
            type=["csv", "txt"],
            key=f"file_status_uploader_{st.session_state.file_uploader_version}"
        )

        if file_uploader is not None:
            fname = file_uploader.name
            try:
                if fname.lower().endswith('.csv'):
                    df_up = parse_uploaded_csv_cached(file_uploader.getvalue())
                    cols = list(df_up.columns)
                    
                    if len(cols) == 1:
                        selected_col = cols[0]
                        st.info(f"Single text column detected: **{selected_col}**")
                    else:
                        selected_col = st.selectbox(
                            "📌 Select the text column",
                            options=cols,
                            index=0,
                            key=f"status_text_column_{fname}_{len(cols)}"
                        )
                    column_values = df_up[selected_col]
                    empty_mask = column_values.isna() | column_values.astype(str).str.strip().eq("")
                    st.session_state.file_empty_count = int(empty_mask.sum())
                    raw_lines = column_values[~empty_mask].astype(str).str.strip().tolist()
                    col_info = selected_col
                else:
                    txt_bytes = file_uploader.getvalue()
                    content = txt_bytes.decode('utf-8', errors='ignore')
                    lines = content.splitlines()
                    st.session_state.file_empty_count = sum(not line.strip() for line in lines)
                    raw_lines = [line.strip() for line in lines if line.strip()]
                    col_info = "TXT lines"

                # Store in session state
                st.session_state.file_texts = raw_lines
                st.session_state.file_filename = fname
                st.session_state.file_col_name = col_info

            except Exception as e:
                st.error(f"File tidak dapat dibaca: {e}")

        # If file data exists in session state
        if st.session_state.file_filename:
            f_len = len(st.session_state.file_texts)
            f_unique = len(set(st.session_state.file_texts))
            f_avg_len = int(np.mean([len(s) for s in st.session_state.file_texts])) if f_len > 0 else 0

            st.markdown(f"""
            <div class="section-card">
                <strong>Nama file:</strong> <code>{escape(st.session_state.file_filename)}</code> ·
                <strong>Kolom teks:</strong> <code>{escape(str(st.session_state.file_col_name))}</code><br>
                <strong>Jumlah dokumen:</strong> {f_len} · <strong>Teks kosong dilewati:</strong> {st.session_state.file_empty_count} ·
                <strong>Teks unik:</strong> {f_unique} · <strong>Rata-rata panjang:</strong> {f_avg_len} karakter
            </div>
            """, unsafe_allow_html=True)

            # Preview up to 10 statuses
            st.markdown(f"##### Pratinjau · {min(10, f_len)} dokumen pertama")
            preview_df = pd.DataFrame({
                'ID': [f"S{i+1}" for i in range(min(10, f_len))],
                'Teks': st.session_state.file_texts[:10]
            })
            st.dataframe(preview_df, use_container_width=True, height=330)

            c_use, c_rst = st.columns([1.5, 1])
            with c_use:
                if st.button("▶ Analisis Dokumen", type="primary", use_container_width=True):
                    st.session_state.file_confirmed = True
                    st.success(f"✓ {f_len} dokumen siap dianalisis.")
            with c_rst:
                if st.button("🗑️ Hapus File", use_container_width=True):
                    st.session_state.file_texts = []
                    st.session_state.file_filename = None
                    st.session_state.file_col_name = None
                    st.session_state.file_confirmed = False
                    st.session_state.file_empty_count = 0
                    st.session_state.file_uploader_version += 1
                    st.rerun()

            active_texts = st.session_state.file_texts

        else:
            st.info("Unggah file CSV atau TXT untuk melihat pratinjau dokumen.")
            active_texts = []

    st.markdown("---")

    # ----------------------------------------------------
    # VALIDASI TEKS
    # ----------------------------------------------------
    empty_count = (
        sum(not str(text).strip() for text in active_texts)
        if st.session_state.text_input_mode == "manual"
        else st.session_state.file_empty_count
    )
    if empty_count:
        st.warning(f"{empty_count} teks kosong akan dilewati.")
    active_texts = [str(text).strip() for text in active_texts if str(text).strip()]
    n_active = len(active_texts)

    if n_active == 0:
        st.info("Belum ada teks.")
        st.stop()
    elif n_active == 1:
        st.warning("Minimal dua teks diperlukan.")
        st.stop()
    else:
        st.success(f"✅ {n_active} teks siap dianalisis.")

    # ----------------------------------------------------
    # 1. PREPROCESSING PIPELINE
    # ----------------------------------------------------
    st.markdown("#### Prapemrosesan Teks")
    st.markdown("""
    <div class="flow-wrapper">
        <div class="flow-node active">TEKS ASLI</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">SERAGAMKAN HURUF</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">HAPUS TANDA BACA</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">TOKENISASI</div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node active">VEKTOR BINER (0/1)</div>
    </div>
    <div class='workflow-desc'>
        Representasi biner: 1 jika kata muncul, selain itu 0. Metode ini tidak menggunakan pembobotan TF-IDF.
    </div>
    """, unsafe_allow_html=True)

    # Reuse cached preprocessing and pairwise results across Streamlit reruns.
    processed_dict, vocab, df_binary, df_pairs = compute_text_similarity_cached(tuple(active_texts))
    prep_records = []
    for idx, (s_id, p_res) in enumerate(processed_dict.items()):
        if idx < 15:
            prep_records.append({
                'ID': s_id,
                'Teks Asli': p_res['original'],
                'Case Folding': p_res['case_folded'],
                'Tanda Baca Dihapus': p_res['punct_removed'],
                'Token': str(p_res['tokens'])
            })

    if n_active > 15:
        st.caption(f"Menampilkan 15 dari {n_active} teks yang diproses.")
    st.dataframe(pd.DataFrame(prep_records), use_container_width=True)

    # ----------------------------------------------------
    # 2. BINARY REPRESENTATION MATRIX
    # ----------------------------------------------------
    st.markdown("#### Representasi Biner")
    st.markdown(f"**Kosakata unik:** `{len(vocab)} kata`")
    
    if n_active <= 15 and len(vocab) <= 30:
        st.dataframe(df_binary, use_container_width=True)
    else:
        st.caption(f"Matriks biner · {n_active} teks × {len(vocab)} kata. Pratinjau 10 teks dan 20 kata.")
        st.dataframe(df_binary.head(10).iloc[:, :20], use_container_width=True)
        with st.expander("📚 Lihat seluruh kosakata"):
            st.write(vocab)

    # ----------------------------------------------------
    # 3. COSINE SIMILARITY RESULTS
    # ----------------------------------------------------
    st.markdown("---")
    st.markdown("#### Hasil Cosine Similarity")
    n_pairs = len(df_pairs)
    category_display = {
        "Sangat Mirip": "Sangat Mirip",
        "Cukup Mirip": "Cukup Mirip",
        "Kurang Mirip": "Kurang Mirip",
        "Tidak Mirip": "Tidak Mirip"
    }

    if not df_pairs.empty:
        top_pair = df_pairs.iloc[0]
        bot_pair = df_pairs.iloc[-1]

        # Summary Cards
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TEKS & PASANGAN</div>
                <div class="metric-value">{n_active} teks</div>
                <div class="metric-sub">{n_pairs:,} pasangan dianalisis</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #10B981;">
                <div class="metric-label">PALING MIRIP</div>
                <div class="metric-value" style="color: #10B981; font-size: 20px;">{top_pair['Pasangan']}</div>
                <div class="metric-sub">{top_pair['Persentase']} ({category_display.get(top_pair['Kategori'], top_pair['Kategori'])})</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m3:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #EF4444;">
                <div class="metric-label">PALING BERBEDA</div>
                <div class="metric-value" style="color: #EF4444; font-size: 20px;">{bot_pair['Pasangan']}</div>
                <div class="metric-sub">{bot_pair['Persentase']} ({category_display.get(bot_pair['Kategori'], bot_pair['Kategori'])})</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Filters for large pairs
        st.markdown("##### Kemiripan Antar-Pasangan")
        
        col_f1, col_f2, col_f3, col_f4 = st.columns([1.5, 1.2, 1.2, 1.2])
        with col_f1:
            search_query = st.text_input("🔎 Cari teks atau pasangan", "")
        with col_f2:
            sort_dir = st.selectbox("Urutkan", ["Tertinggi ke terendah", "Terendah ke tertinggi"])
        with col_f3:
            category_labels = {
                "Semua kategori": None,
                "Sangat mirip": "Sangat Mirip",
                "Cukup mirip": "Cukup Mirip",
                "Kurang mirip": "Kurang Mirip",
                "Tidak mirip": "Tidak Mirip"
            }
            cat_filter = st.selectbox("Tingkat Kemiripan", list(category_labels))
        with col_f4:
            limit_opt = st.selectbox("Jumlah baris", ["Top 15", "Top 30", "Top 50", "Top 100", "Semua pasangan"])

        # Apply filtering
        filtered_df = df_pairs.copy()
        if search_query.strip():
            sq = search_query.strip().upper()
            filtered_df = filtered_df[
                filtered_df['Pasangan'].str.upper().str.contains(sq, regex=False) |
                filtered_df['Status 1'].str.upper().str.contains(sq, regex=False) |
                filtered_df['Status 2'].str.upper().str.contains(sq, regex=False)
            ]
        if category_labels[cat_filter] is not None:
            filtered_df = filtered_df[filtered_df['Kategori'] == category_labels[cat_filter]]

        if "Terendah" in sort_dir:
            filtered_df = filtered_df.sort_values(by='Similarity', ascending=True)
        else:
            filtered_df = filtered_df.sort_values(by='Similarity', ascending=False)

        # Limit row display
        limit_map = {"Top 15": 15, "Top 30": 30, "Top 50": 50, "Top 100": 100, "Semua pasangan": len(filtered_df)}
        max_rows = limit_map.get(limit_opt, 15)
        display_df = filtered_df.head(max_rows)

        display_columns = {
            'Pasangan': 'Pasangan Teks', 'Status 1': 'Teks 1', 'Status 2': 'Teks 2',
            'Dot Product': 'Hasil Kali Titik', 'Similarity': 'Nilai Similarity',
            'Persentase': 'Persentase', 'Kategori': 'Tingkat Kemiripan', 'Kata Irisan': 'Kata Irisan'
        }
        display_table = display_df.rename(columns=display_columns).copy()
        display_table['Tingkat Kemiripan'] = display_table['Tingkat Kemiripan'].replace(category_display)
        st.dataframe(display_table, use_container_width=True, height=360)
        st.caption(f"Menampilkan {len(display_df)} dari {len(filtered_df)} pasangan hasil filter.")

        # ----------------------------------------------------
        # 4. VISUALISASI
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown("##### Sebaran Nilai Similarity")

        if n_active <= 10:
            fig, ax = plt.subplots(figsize=(7, max(2.5, len(df_pairs) * 0.35)))
            plot_pairs = df_pairs.head(15)
            color_map = {'Sangat Mirip': '#10B981', 'Cukup Mirip': '#2563EB', 'Kurang Mirip': '#F59E0B', 'Tidak Mirip': '#EF4444'}
            cols_bar = [color_map.get(k, '#2563EB') for k in plot_pairs['Kategori']]
            bars = ax.barh(plot_pairs['Pasangan'], plot_pairs['Similarity'], color=cols_bar)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("Nilai Cosine Similarity")
            ax.set_title("Nilai similarity antar-pasangan", fontsize=10, fontweight='bold')
            for b in bars:
                w = b.get_width()
                ax.text(w + 0.02, b.get_y() + b.get_height()/2, f"{w:.4f} ({w*100:.1f}%)", va='center', fontsize=8.5, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(f"{n_active} teks · {n_pairs:,} pasangan. Grafik terfokus ditampilkan di bawah.")
            tab_v1, tab_v2, tab_v3 = st.tabs(["10 Teratas · Paling Mirip", "10 Teratas · Paling Berbeda", "Sebaran Kategori"])
            
            with tab_v1:
                top_10 = df_pairs.head(10)
                fig, ax = plt.subplots(figsize=(7, 3.2))
                bars = ax.barh(top_10['Pasangan'], top_10['Similarity'], color='#10B981')
                ax.set_xlim(0, 1.05)
                ax.set_title("10 Pasangan Paling Mirip", fontsize=10, fontweight='bold')
                for b in bars:
                    w = b.get_width()
                    ax.text(w + 0.02, b.get_y() + b.get_height()/2, f"{w:.4f} ({w*100:.1f}%)", va='center', fontsize=8, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with tab_v2:
                bot_10 = df_pairs.tail(10).sort_values(by='Similarity', ascending=True)
                fig, ax = plt.subplots(figsize=(7, 3.2))
                bars = ax.barh(bot_10['Pasangan'], bot_10['Similarity'], color='#EF4444')
                ax.set_xlim(0, 1.05)
                ax.set_title("10 Pasangan Paling Berbeda", fontsize=10, fontweight='bold')
                for b in bars:
                    w = b.get_width()
                    ax.text(w + 0.02, b.get_y() + b.get_height()/2, f"{w:.4f} ({w*100:.1f}%)", va='center', fontsize=8, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with tab_v3:
                cat_counts = df_pairs['Kategori'].value_counts()
                cat_counts.index = [category_display.get(category, category) for category in cat_counts.index]
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(x=cat_counts.index, y=cat_counts.values, palette=['#10B981', '#2563EB', '#F59E0B', '#EF4444'][:len(cat_counts)], ax=ax)
                ax.set_title("Kategori Kemiripan", fontsize=10, fontweight='bold')
                ax.set_ylabel("Jumlah pasangan")
                for p in ax.patches:
                    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8.5, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ----------------------------------------------------
        # 5. DETAIL PERHITUNGAN LANGKAH-DEMI-LANGKAH
        # ----------------------------------------------------
        st.markdown("---")
        with st.expander("🔍 Lihat Detail Perhitungan Pasangan"):
            st.markdown("Pilih pasangan untuk melihat vektor biner, kata yang sama, dot product, dan magnitude.")
            
            pair_choice_list = list(df_pairs['Pasangan'])
            dropdown_options = pair_choice_list[:200]
            sel_pair = st.selectbox("Pilih pasangan teks", options=dropdown_options, index=0)
            
            if sel_pair:
                parts = sel_pair.split(" - ")
                if len(parts) == 2:
                    l1, l2 = parts[0].strip(), parts[1].strip()
                    det = compute_single_pair_detail(l1, l2, df_binary, vocab)
                    
                    if det:
                        c_va, c_vb = st.columns(2)
                        with c_va:
                            st.markdown(f"**Vektor {l1}:**")
                            st.code(str(det['vec_a']))
                        with c_vb:
                            st.markdown(f"**Vektor {l2}:**")
                            st.code(str(det['vec_b']))

                        st.markdown(f"""
                        - **Kata yang sama:** `{', '.join(det['common_terms']) if det['common_terms'] else 'Tidak ada'}`
                        - **Hasil kali titik (A · B):** `{' + '.join(['1×1' for _ in det['common_terms']]) if det['common_terms'] else '0'} = {int(det['dot_product'])}`
                        - **Norma ||{l1}||:** `sqrt({sum(det['vec_a'])}) = {det['mag_a']:.4f}`
                        - **Norma ||{l2}||:** `sqrt({sum(det['vec_b'])}) = {det['mag_b']:.4f}`
                        """)

                        if det['mag_a'] * det['mag_b'] > 0:
                            st.latex(rf"\text{{Cos}}({l1}, {l2}) = \frac{{{int(det['dot_product'])}}}{{{det['mag_a']:.4f} \times {det['mag_b']:.4f}}} = \frac{{{int(det['dot_product'])}}}{{{(det['mag_a'] * det['mag_b']):.4f}}} = {det['similarity']:.4f}")
                        else:
                            st.latex(rf"\text{{Cos}}({l1}, {l2}) = 0.0000")

                        st.markdown(f"**Hasil:** `{det['percentage']:.2f}%` | **Tingkat Kemiripan:** `{category_display.get(det['kategori'], det['kategori'])}`")


# ==========================================
# 8. PAGE: TENTANG SISTEM
# ==========================================
elif menu_choice == "ℹ️ Tentang Sistem":
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">INFORMASI PROJECT</div>
        <h1>Tentang Sistem</h1>
        <p>Workspace untuk prapemrosesan dataset yang aman dari Data Leakage dan analisis Cosine Similarity teks.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="about-grid">
        <div class="about-card"><div class="feature-icon">◎</div><div class="about-title">Tujuan Sistem</div><div class="about-desc">Menyiapkan dataset untuk analisis melalui tahapan prapemrosesan yang konsisten dan mudah ditinjau.</div></div>
        <div class="about-card"><div class="feature-icon">⌘</div><div class="about-title">Teknologi</div><div class="about-desc">Python, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, dan Scikit-Learn Pipeline.</div></div>
        <div class="about-card"><div class="feature-icon">▦</div><div class="about-title">Prapemrosesan</div><div class="about-desc">Imputasi, pembatasan pencilan berbasis IQR, scaling numerik, One-Hot Encoding, dan Ordinal Encoding.</div></div>
        <div class="about-card"><div class="feature-icon">✦</div><div class="about-title">Feature Engineering</div><div class="about-desc">Membuat Log Income, Expense to Income Ratio, Net Savings, dan Age Group jika kolom sumber tersedia.</div></div>
        <div class="about-card"><div class="feature-icon">✓</div><div class="about-title">Pencegahan Data Leakage</div><div class="about-desc">Fit prapemrosesan hanya pada baris training. Data testing ditransformasi tanpa fit ulang.</div></div>
        <div class="about-card"><div class="feature-icon">01</div><div class="about-title">Cosine Similarity</div><div class="about-desc">Membandingkan teks dengan penyeragaman huruf, tokenisasi, vektor kehadiran biner, dan Cosine Similarity.</div></div>
    </div>
    """, unsafe_allow_html=True)
