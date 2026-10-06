import streamlit as st
import pandas as pd
import numpy as np
import io


def apply_custom_css():
    """Apply the shared visual system for the analytics dashboard."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --canvas: #f7f8fb;
        --surface: #ffffff;
        --ink: #1c2738;
        --muted: #718096;
        --line: #e5e8ef;
        --blue: #3568c9;
        --blue-dark: #244b91;
        --violet: #7166c7;
        --green: #14866d;
        --amber: #b87919;
        --red: #c34e61;
        --shadow: 0 10px 28px rgba(27, 43, 72, 0.06);
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp, [data-testid="stAppViewContainer"] > .main { background: var(--canvas); }
    [data-testid="stHeader"] { height: .35rem; background: transparent; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { max-width: 1460px; padding-top: 1.7rem; padding-bottom: 3rem; }
    h1, h2, h3, h4, h5 { color: var(--ink); font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    h1 { font-size: 2.15rem; line-height: 1.14; }
    h2 { font-size: 1.65rem; }
    h3 { font-size: 1.35rem; }
    [data-testid="stMarkdownContainer"] p { color: #586982; line-height: 1.65; }

    [data-testid="stSidebar"] { background: #152238; border-right: 1px solid #26344b; }
    [data-testid="stSidebar"] > div:first-child { padding: 1.1rem 1rem 1.5rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: .35rem; }
    .brand-lockup { padding: .55rem .55rem 1.4rem; }
    .brand-name { font: 700 1.45rem 'Space Grotesk', sans-serif; color: #ffffff; }
    .brand-subtitle { color: #aab7cb; font-size: .78rem; margin-top: .2rem; }
    .nav-caption, .eyebrow { color: var(--blue); font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .st-key-main_navigation [role="radiogroup"] { gap: .28rem; }
    .st-key-main_navigation label[data-testid="stRadioOption"] { padding: .55rem .7rem; border: 1px solid transparent; border-radius: 10px; background: transparent; color: #d1d9e7; transition: background .16s ease, border-color .16s ease, transform .16s ease; }
    .st-key-main_navigation label[data-testid="stRadioOption"]:hover { background: #202f47; border-color: #2d3e59; transform: translateX(2px); }
    .st-key-main_navigation label[data-testid="stRadioOption"][data-selected="true"] { background: #263a5b; border-color: #3a527a; color: #ffffff; font-weight: 700; }
    .st-key-main_navigation label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
    .nav-hint { margin: .75rem 0 1.15rem; padding: .7rem .75rem; border-radius: 10px; background: #202f47; color: #c1ccdc; font-size: .76rem; line-height: 1.45; }
    .dataset-status { margin-top: 1.4rem; padding: 1rem; background: #202f47; border: 1px solid #31425d; border-radius: 14px; box-shadow: 0 6px 18px rgba(0,0,0,.12); }
    .dataset-status-title { color: #aab7cb; font-size: .7rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; margin-bottom: .7rem; }
    .status-pill { display: inline-flex; align-items: center; gap: .42rem; padding: .28rem .6rem; border-radius: 999px; font-size: .7rem; font-weight: 700; }
    .status-pill.active { background: #e4f5ef; color: #14735e; }
    .status-pill.empty { background: #f0f2f7; color: #68758b; }
    .dataset-file { font-size: .84rem; font-weight: 700; color: #ffffff; margin: .65rem 0 .25rem; overflow-wrap: anywhere; }
    .dataset-meta { color: #bdc8d7; font-size: .75rem; line-height: 1.6; }

    .page-heading { margin: .25rem 0 1.35rem; }
    .page-heading .eyebrow { margin-bottom: .45rem; }
    .page-heading h1, .page-heading h2 { margin: 0; }
    .page-heading p { max-width: 760px; margin: .5rem 0 0; }
    .section-heading { margin: 1.75rem 0 .75rem; }
    .section-heading h2 { margin: .3rem 0 0; font-size: 1.35rem; }
    .hero { position: relative; overflow: hidden; display: flex; justify-content: space-between; align-items: center; gap: 2rem; min-height: 225px; padding: 2.25rem 2.5rem; margin-bottom: 1.35rem; color: white; border: 1px solid #27456f; border-radius: 22px; background: linear-gradient(110deg, #18304f 0%, #285590 100%); box-shadow: 0 14px 32px rgba(27,43,72,.13); }
    .hero::after { content: ''; position: absolute; inset: 0 0 0 68%; background: linear-gradient(135deg, rgba(255,255,255,.08), transparent 75%); pointer-events: none; }
    .hero-copy { position: relative; z-index: 1; max-width: 760px; }
    .hero .eyebrow { color: #c6d2ff; }
    .hero h1 { max-width: 690px; color: white; font-size: 2.55rem; margin: .55rem 0; }
    .hero p { max-width: 680px; color: #e4eaff; font-size: 1rem; margin: 0; }
    .hero-mark { position: relative; z-index: 1; display: grid; place-items: center; flex: 0 0 116px; width: 116px; aspect-ratio: 1; border: 1px solid rgba(255,255,255,.26); border-radius: 28px; color: white; background: rgba(255,255,255,.1); font-size: 2.7rem; }
    .empty-state { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(230px, .6fr); align-items: center; gap: 2rem; padding: 2rem 2.1rem; margin: .75rem 0 1.4rem; border: 1px solid var(--line); border-radius: 18px; background: white; box-shadow: var(--shadow); }
    .empty-state h2 { margin: .3rem 0 .5rem; font-size: 1.65rem; }
    .empty-state p { margin: 0; max-width: 600px; }
    .empty-art { min-height: 142px; display: grid; place-items: center; border-radius: 15px; background: linear-gradient(135deg, #edf2ff, #f0edff); color: var(--blue); font-size: 3.25rem; }
    .empty-actions { display: flex; gap: .7rem; flex-wrap: wrap; margin-top: 1.15rem; }

    .metric-card { min-height: 136px; background: var(--surface); border: 1px solid var(--line); border-radius: 15px; padding: 1.1rem 1.2rem; box-shadow: 0 6px 20px rgba(32,55,104,.045); transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
    .metric-card:hover { transform: translateY(-3px); border-color: #cbd5fa; box-shadow: var(--shadow); }
    .metric-label { color: var(--muted); font-size: .7rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .metric-value { color: var(--ink); font: 700 1.9rem 'Space Grotesk', sans-serif; line-height: 1.2; margin-top: .45rem; }
    .metric-sub { color: #8b97ab; font-size: .76rem; margin-top: .4rem; }
    .section-card, [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line) !important; border-radius: 15px !important; box-shadow: 0 6px 20px rgba(32,55,104,.04); }
    .section-title { color: var(--ink); font: 700 1.03rem 'Space Grotesk', sans-serif; margin-bottom: .8rem; }
    .purpose-card { padding: 1rem 1.15rem; border: 1px solid #dbe5ff; border-left: 4px solid var(--blue); border-radius: 12px; background: #f7f9ff; }
    .purpose-title { color: var(--blue-dark); font-weight: 700; font-size: .92rem; }
    .purpose-text { color: #61718c; font-size: .86rem; line-height: 1.6; margin-top: .35rem; }
    .workflow-grid, .feature-grid, .about-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: .8rem 0 1.2rem; }
    .workflow-card, .feature-card, .about-card { min-height: 146px; padding: 1rem; border: 1px solid var(--line); border-radius: 13px; background: white; box-shadow: 0 5px 16px rgba(32,55,104,.035); }
    .workflow-icon, .feature-icon { width: 36px; height: 36px; display: grid; place-items: center; margin-bottom: .7rem; border-radius: 10px; background: #eaf0ff; color: var(--blue); font-size: 1.05rem; }
    .workflow-title, .feature-title, .about-title { color: var(--ink); font-weight: 700; font-size: .88rem; }
    .workflow-desc, .feature-desc, .about-desc { color: var(--muted); font-size: .78rem; line-height: 1.55; margin-top: .35rem; }
    .feature-formula { display: inline-block; margin-top: .7rem; padding: .3rem .48rem; color: #4b54a7; background: #f0efff; border-radius: 7px; font: 600 .72rem 'DM Sans', sans-serif; }
    .flow-wrapper { display: flex; flex-wrap: wrap; gap: .55rem; align-items: center; padding: 1rem; margin-bottom: 1rem; border: 1px solid var(--line); border-radius: 14px; background: white; }
    .flow-node { padding: .55rem .75rem; color: #455574; background: #f7f8fc; border: 1px solid var(--line); border-radius: 9px; font-size: .73rem; font-weight: 700; text-align: center; }
    .flow-node.active { color: var(--blue-dark); background: #eaf0ff; border-color: #cbd7ff; }
    .flow-arrow { color: #9ba8bd; font-size: .9rem; font-weight: 700; }
    .leakage-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .9rem; margin: 1rem 0; }
    .leakage-lane { padding: 1.1rem; border: 1px solid var(--line); border-radius: 15px; background: white; }
    .lane-heading { color: var(--muted); font-size: .72rem; font-weight: 700; letter-spacing: .1em; margin-bottom: .8rem; }
    .lane-dot { display: inline-block; width: 8px; height: 8px; margin-right: .35rem; border-radius: 50%; background: var(--blue); }
    .test-lane .lane-dot { background: var(--violet); }
    .lane-step { padding: .62rem .7rem; color: #3c4d69; background: #f7f8fc; border: 1px solid var(--line); border-radius: 9px; font-size: .78rem; font-weight: 700; text-align: center; }
    .lane-step.emphasized { color: var(--blue-dark); background: #eef2ff; border-color: #ccd7ff; }
    .lane-step.muted-step { color: #66758e; }
    .lane-arrow { padding: .28rem; color: #9ba8bd; text-align: center; }
    .no-refit { margin-top: .75rem; color: #694caf; font-size: .69rem; font-weight: 800; letter-spacing: .1em; text-align: center; }
    .leakage-shield-box { padding: 1.15rem; margin-bottom: 1rem; border: 1px solid #bfe6d9; border-radius: 14px; background: #f0faf6; }
    .leakage-title { color: #14745e; font-weight: 700; font-size: 1rem; }
    .leakage-desc { color: #527d70; font-size: .84rem; line-height: 1.6; margin-top: .35rem; }
    .concept-item { padding: .7rem .8rem; margin-bottom: .55rem; border: 1px solid var(--line); border-radius: 10px; background: white; }
    .concept-title { color: var(--ink); font-size: .83rem; font-weight: 700; }
    .concept-desc { color: var(--muted); font-size: .76rem; line-height: 1.5; }
    .badge { display: inline-flex; align-items: center; padding: .25rem .55rem; border-radius: 999px; font-size: .7rem; font-weight: 700; }
    .badge-success { color: #14745e; background: #e4f5ef; border: 1px solid #caebdf; }
    .badge-info { color: #3455bb; background: #eaf0ff; border: 1px solid #d5dfff; }
    .badge-warning { color: #9b6819; background: #fff4dc; border: 1px solid #f4e1b7; }
    .badge-danger { color: #a94355; background: #fff0f2; border: 1px solid #f3d5db; }

    .stButton > button, .stDownloadButton > button { min-height: 2.55rem; padding: .55rem .95rem; border-radius: 10px; border: 1px solid #d7dfed; color: #34445f; background: white; font-weight: 700; transition: all .16s ease; }
    [data-testid="stSidebar"] .stButton > button { color: #e5ebf4; background: #202f47; border-color: #364863; }
    [data-testid="stSidebar"] .stButton > button:hover { color: white; background: #2b3e5c; border-color: #60769a; }
    .stButton > button:hover, .stDownloadButton > button:hover { color: var(--blue-dark); border-color: #b9c8ff; background: #f7f9ff; box-shadow: 0 5px 14px rgba(49,78,172,.1); }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { color: white; border-color: var(--blue); background: linear-gradient(105deg, var(--blue), var(--violet)); }
    .stButton > button[kind="primary"]:hover { color: white; filter: brightness(1.04); box-shadow: 0 7px 18px rgba(66,103,232,.23); }
    [data-testid="stFileUploader"] section { padding: 1rem; border: 1px dashed #b9c7e4; border-radius: 13px; background: #f8faff; }
    [data-testid="stFileUploader"] section:hover { border-color: var(--blue); background: #f2f5ff; }
    [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p { display: none; }
    [data-testid="stFileUploaderDropzone"] button::after { content: 'Pilih file'; font-size: .8rem; }
    [data-testid="stFileUploaderDropzoneInstructions"] span { display: none; }
    [data-testid="stFileUploaderDropzoneInstructions"]::after { content: 'Maksimum 200 MB per file'; color: var(--muted); font-size: .75rem; }
    [data-testid="stTabs"] [role="tablist"] { gap: .4rem; border-bottom: 1px solid var(--line); }
    [data-testid="stTabs"] [role="tablist"] { overflow-x: auto; flex-wrap: nowrap; }
    [data-testid="stTabs"] button[role="tab"] { padding: .65rem .85rem; color: #71809a; font-size: .82rem; font-weight: 700; }
    [data-testid="stTabs"] button[role="tab"] { flex: 0 0 auto; white-space: nowrap; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--blue-dark); }
    [data-testid="stDataFrame"], [data-testid="stTable"] { overflow: hidden; border: 1px solid var(--line); border-radius: 12px; background: white; }
    [data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--line); }
    div[data-testid="stMetric"] { padding: .85rem 1rem; background: white; border: 1px solid var(--line); border-radius: 13px; box-shadow: 0 4px 14px rgba(32,55,104,.035); }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .72rem; font-weight: 700; text-transform: uppercase; }
    div[data-testid="stMetricValue"] { color: var(--ink); font: 700 1.55rem 'Space Grotesk', sans-serif; }
    .stRadio [role="radiogroup"] { gap: .45rem; }
    .stRadio [role="radiogroup"] label { padding: .45rem .75rem; border: 1px solid var(--line); border-radius: 10px; background: white; }
    .stRadio [role="radiogroup"] label[data-checked="true"] { color: var(--blue-dark); border-color: #c7d3ff; background: #edf1ff; }
    .st-key-similarity_input_mode [role="radiogroup"] { display: flex; width: fit-content; padding: .22rem; border: 1px solid var(--line); border-radius: 12px; background: white; }
    .st-key-similarity_input_mode label[data-testid="stRadioOption"] { padding: .55rem .85rem; border: 1px solid transparent; border-radius: 9px; background: transparent; }
    .st-key-similarity_input_mode label[data-testid="stRadioOption"][data-selected="true"] { color: var(--blue-dark); border-color: #d3dcff; background: #edf1ff; }
    .st-key-similarity_input_mode label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
    hr { border-color: var(--line); }

    @media (max-width: 900px) {
        .block-container { padding: 1.25rem 1rem 2rem; }
        .hero { min-height: 195px; padding: 1.5rem; }
        .hero h1 { font-size: 2rem; }
        .workflow-grid, .feature-grid, .about-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .leakage-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 600px) {
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { width: 100% !important; min-width: 0 !important; max-width: 100vw !important; overflow-x: hidden !important; }
        .block-container, [data-testid="stMainBlockContainer"] { width: 100% !important; min-width: 0 !important; max-width: 100% !important; box-sizing: border-box; }
        .hero { align-items: flex-start; min-height: 0; padding: 1.25rem; }
        .hero h1 { font-size: 1.7rem; }
        .hero-mark { flex-basis: 56px; width: 56px; border-radius: 15px; font-size: 1.5rem; }
        .empty-state { grid-template-columns: 1fr; padding: 1.2rem; }
        .empty-art { min-height: 90px; }
        .workflow-grid, .feature-grid, .about-grid { grid-template-columns: 1fr; }
    }
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    """Initialize persistent dataset and text state for Streamlit."""
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = "Belum Ada Dataset"
    if 'numerical_cols' not in st.session_state:
        st.session_state.numerical_cols = []
    if 'nominal_cols' not in st.session_state:
        st.session_state.nominal_cols = []
    if 'ordinal_cols' not in st.session_state:
        st.session_state.ordinal_cols = []
    if 'ordinal_categories' not in st.session_state:
        st.session_state.ordinal_categories = {}
    if 'pipeline_result' not in st.session_state:
        st.session_state.pipeline_result = None
    if 'dataset_upload_digest' not in st.session_state:
        st.session_state.dataset_upload_digest = None
    if 'dataset_uploader_version' not in st.session_state:
        st.session_state.dataset_uploader_version = 0
    if 'fe_dataset' not in st.session_state:
        st.session_state.fe_dataset = None
    if 'fe_summary' not in st.session_state:
        st.session_state.fe_summary = None
    if 'fe_config' not in st.session_state:
        st.session_state.fe_config = {
            'enable_log_income': True,
            'enable_expense_ratio': True,
            'enable_net_savings': True,
            'enable_age_binning': True,
            'income_col': 'Income',
            'expense_col': 'Monthly_Expense',
            'age_col': 'Age'
        }
    if 'text_list' not in st.session_state:
        st.session_state.text_list = []
    if 'text_input_mode' not in st.session_state:
        st.session_state.text_input_mode = "manual"
    if 'manual_texts' not in st.session_state:
        st.session_state.manual_texts = []
    if 'manual_text_ids' not in st.session_state:
        st.session_state.manual_text_ids = list(range(1, len(st.session_state.manual_texts) + 1))
    if 'manual_text_next_id' not in st.session_state:
        st.session_state.manual_text_next_id = max(st.session_state.manual_text_ids, default=0) + 1
    if 'file_texts' not in st.session_state:
        st.session_state.file_texts = []
    if 'file_filename' not in st.session_state:
        st.session_state.file_filename = None
    if 'file_col_name' not in st.session_state:
        st.session_state.file_col_name = None
    if 'file_confirmed' not in st.session_state:
        st.session_state.file_confirmed = False
    if 'file_empty_count' not in st.session_state:
        st.session_state.file_empty_count = 0
    if 'file_uploader_version' not in st.session_state:
        st.session_state.file_uploader_version = 0


@st.cache_data
def load_sample_dataset_cached(filepath: str) -> pd.DataFrame:
    """Caches loading of the CSV sample dataset."""
    return pd.read_csv(filepath)


@st.cache_data
def parse_uploaded_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    """Caches parsing of uploaded CSV bytes."""
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data
def compute_text_similarity_cached(texts: tuple):
    """Caches preprocessing and binary cosine analysis for an ordered text set."""
    from modules.similarity import (
        preprocess_text_pipeline,
        build_vocabulary,
        build_binary_matrix,
        compute_all_pairs_similarity
    )

    processed_texts = {
        f"S{index + 1}": preprocess_text_pipeline(text)
        for index, text in enumerate(texts)
    }
    vocabulary = build_vocabulary(processed_texts)
    binary_matrix = build_binary_matrix(processed_texts, vocabulary)
    pairs, _ = compute_all_pairs_similarity(binary_matrix, vocabulary)
    return processed_texts, vocabulary, binary_matrix, pairs


@st.cache_data
def compute_data_summary_cached(df: pd.DataFrame):
    """Caches general dataset size and missing count."""
    return len(df), df.shape[1], int(df.isna().sum().sum())


@st.cache_data
def compute_missing_table_cached(df: pd.DataFrame) -> pd.DataFrame:
    """Caches missing value detection per column."""
    records = []
    row_count = len(df)
    for column in df.columns:
        missing = int(df[column].isna().sum())
        records.append({
            'Atribut': column,
            'Missing': missing,
            'Persentase': f"{(missing / row_count) * 100:.2f}%" if row_count > 0 else "0.00%",
            'Nilai_Persen': (missing / row_count) * 100 if row_count > 0 else 0
        })
    return pd.DataFrame(records)


@st.cache_data
def compute_outliers_table_cached(df: pd.DataFrame, num_cols: tuple, factor: float = 1.5) -> pd.DataFrame:
    """Caches IQR outlier detection for numerical columns."""
    from modules.outlier import detect_outliers_iqr
    return detect_outliers_iqr(df, list(num_cols), factor=factor)


@st.cache_data
def compute_descriptive_stats_cached(df: pd.DataFrame, num_cols: tuple) -> pd.DataFrame:
    """Caches numerical descriptive statistics."""
    valid_cols = [column for column in num_cols if column in df.columns]
    if not valid_cols:
        return pd.DataFrame()
    desc = df[valid_cols].describe().T
    ordered_cols = [column for column in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'] if column in desc.columns]
    return desc[ordered_cols].round(2)


def auto_detect_column_types(df: pd.DataFrame):
    """Classify columns as numerical, nominal, or ordinal."""
    numerical_cols = []
    nominal_cols = []
    ordinal_cols = []
    ordinal_categories = {}
    known_ordinals = {
        'satisfaction': ['Very Unsatisfied', 'Unsatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'],
        'education_level': ['High School', 'Bachelor', 'Master', 'Doctorate'],
        'service_rating': ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
        'rating': ['1', '2', '3', '4', '5'],
        'grade': ['D', 'C', 'B', 'A']
    }

    for column in df.columns:
        column_lower = column.lower().strip()
        matched_order = next((values for key, values in known_ordinals.items() if key in column_lower), None)
        if matched_order:
            ordinal_cols.append(column)
            unique_values = list(df[column].dropna().unique())
            ordered_values = [value for value in matched_order if value in unique_values]
            ordered_values.extend(value for value in unique_values if value not in ordered_values)
            ordinal_categories[column] = ordered_values if ordered_values else unique_values
        elif pd.api.types.is_numeric_dtype(df[column]):
            numerical_cols.append(column)
        elif any(token in column_lower for token in ('satisfaction', 'rating', 'level')):
            ordinal_cols.append(column)
            ordinal_categories[column] = sorted(str(value) for value in df[column].dropna().unique())
        else:
            nominal_cols.append(column)

    return numerical_cols, nominal_cols, ordinal_cols, ordinal_categories


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Encode a dataframe as UTF-8 CSV bytes for Streamlit downloads."""
    return df.to_csv(index=False).encode('utf-8')
