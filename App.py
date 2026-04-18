"""
PAT.ai 
Hackathon (PS 3)
"""

from typing import Optional, Dict, Any, List
import re
import os
import hashlib
import base64
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.impute import SimpleImputer
import faiss
from rapidfuzz import fuzz

warnings.filterwarnings('ignore')

# ---------------------- Background Images ----------------------
def get_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

dark_bg = get_base64("your_dark_bg.png")
light_bg = get_base64("your_light_bg.png")

# ---------------------- Session State ----------------------
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'chat_started' not in st.session_state:
    st.session_state['chat_started'] = False
if 'has_merged' not in st.session_state:
    st.session_state['has_merged'] = False
if 'uploaded_filename' not in st.session_state:
    st.session_state['uploaded_filename'] = None

# ---------------------- Prompt Parsing ----------------------
ACTION_MAP = {
    'mean': ['mean', 'average', 'avg'],
    'describe': ['describe', 'summary', 'summary statistics', 'tell me about the data'],
    'head': ['head', 'show head', 'first rows', 'preview the data'],
    'tail': ['tail', 'last rows'],
    'dropna': ['dropna', 'drop na', 'drop missing'],
    'fillna': ['fillna', 'fill missing', 'impute'],
    'histogram': ['histogram', 'hist', 'distribution'],
    'barchart': ['bar chart', 'bar'],
    'heatmap': ['heatmap', 'correlation heatmap'],
    'pie': ['pie'],
    'scatter': ['scatter', 'scatter plot'],
    'count': ['count', 'value counts'],
    'corr': ['correlation', 'corr'],
    'rows': ['rows', 'row count', 'how many rows'],
    'columns': ['columns', 'col count', 'how many columns'],
    'dtypes': ['datatypes', 'dtypes', 'column types'],
    'data_quality': ['data quality', 'check quality', 'missing values', 'outliers'],
    'feature_types': ['categorical', 'numerical', 'feature types'],
    'duplicate': ['duplicate', 'duplicates', 'multilingual duplicates', 'detect duplicates', 'find duplicates'],
    'download': ['download', 'export', 'save', 'download csv'],
    'insights': ['insights', 'key insights', 'analyze'], 
    'hello': ['hello', 'how are you', 'heyy!', 'hi'] 
}

INVERSE_ACTION = {}
for k, vs in ACTION_MAP.items():
    for v in vs:
        INVERSE_ACTION[v] = k

def detect_actions(text: str) -> List[str]:
    text_low = text.lower()
    actions = []
    for phrase, action in INVERSE_ACTION.items():
        if phrase in text_low and action not in actions: actions.append(action)
    for k in ACTION_MAP.keys():
        if k in text_low and k not in actions: actions.append(k)
    return actions

def extract_column_names(text: str, df: pd.DataFrame) -> List[str]:
    if df is None: return []
    cols = list(df.columns.astype(str))
    found = []
    for col in cols:
        if re.search(rf"\b{re.escape(col)}\b", text, flags=re.IGNORECASE): found.append(col)
    if found: return found
    return list(dict.fromkeys(found))

# ---------------------- ML Model Loading & Caching ----------------------
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('LaBSE')
        return model, "LaBSE"
    except ImportError:
        return None, "fallback"

@st.cache_resource(show_spinner=False)
def load_catboost_model():
    """Loads the Stage-2 Re-ranking CatBoost Model if it exists."""
    if os.path.exists("catboost_duplicate_model.cbm"):
        try:
            from catboost import CatBoostClassifier
            cb = CatBoostClassifier()
            cb.load_model("catboost_duplicate_model.cbm")
            return cb
        except Exception as e:
            print(f"Failed to load CatBoost: {e}")
    return None

CACHE_FILE = "embeddings_cache.npy"
HASH_FILE  = "embeddings_hash.txt"

def get_dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()

def get_embeddings(texts: List[str], model, model_type: str, df: pd.DataFrame = None):
    if df is not None and os.path.exists(CACHE_FILE) and os.path.exists(HASH_FILE):
        with open(HASH_FILE, 'r') as f:
            if get_dataset_hash(df) == f.read().strip():
                st.toast("⚡ Embeddings loaded from cache — instant!", icon="✅")
                return np.load(CACHE_FILE)

    if model_type == "fallback" or model is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        embeddings = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=512).fit_transform(texts).toarray()
    else:
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=64)

    np.save(CACHE_FILE, embeddings)
    if df is not None:
        with open(HASH_FILE, 'w') as f:
            f.write(get_dataset_hash(df))
    return embeddings

# ---------------------- TWO-STAGE PIPELINE: FAISS + CATBOOST ----------------------
def detect_duplicates(df: pd.DataFrame, text_col: str, base_threshold: float, model, model_type: str) -> pd.DataFrame:
    texts = df[text_col].astype(str).tolist()
    cb_model = load_catboost_model()
    has_complex_metadata = all(col in df.columns for col in ['category', 'region', 'created_at', 'name'])

    with st.spinner(" Generating multilingual embeddings..."):
        embeddings = get_embeddings(texts, model, model_type, df=df)

    with st.spinner(" Indexing with FAISS for scalable search..."):
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        
        index = faiss.IndexFlatIP(embeddings_np.shape[1]) 
        index.add(embeddings_np)
        
        k = min(10, len(df))
        distances, indices = index.search(embeddings_np, k)

    pairs = []
    seen = set()
    n = len(df)
    
    if cb_model and has_complex_metadata:
        st.toast(" PAT is using the Stage-2 CatBoost Re-ranker!", icon="🚀")
    
    for i in range(n):
        for rank, j in enumerate(indices[i]):
            if i == j: continue
            score = float(distances[i][rank])
            
            # --- STAGE 2 CATBOOST LOGIC ---
            if cb_model and has_complex_metadata:
                if score >= 0.80:
                    pair_key = tuple(sorted((i, int(j))))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        
                        row_i, row_j = df.iloc[i], df.iloc[j]
                        
                        # Build the exact features we trained on
                        name_fuzz = fuzz.ratio(str(row_i.get('name', '')), str(row_j.get('name', ''))) / 100.0
                        cat_A = str(row_i.get('category', 'Unknown'))
                        cat_B = str(row_j.get('category', 'Unknown'))
                        region_A = str(row_i.get('region', 'Unknown'))
                        region_B = str(row_j.get('region', 'Unknown'))
                        
                        try:
                            if pd.notnull(row_i.get('created_at')) and pd.notnull(row_j.get('created_at')):
                                t1 = pd.to_datetime(row_i['created_at'])
                                t2 = pd.to_datetime(row_j['created_at'])
                                time_diff = abs((t1 - t2).total_seconds()) / 3600.0
                            else: time_diff = 9999.0
                        except: time_diff = 9999.0
                        
                        features = [score, name_fuzz, cat_A, cat_B, region_A, region_B, time_diff]
                        
                        # CatBoost final decision
                        prediction = cb_model.predict([features])[0]
                        
                        if prediction == 1:
                            lang_i = str(row_i.get('language', '')).strip()
                            lang_j = str(row_j.get('language', '')).strip()
                            pairs.append({
                                'Record A ID': row_i.get('id', i),
                                'Record A Text': row_i[text_col],
                                'Language A': lang_i,
                                'Record B ID': row_j.get('id', j),
                                'Record B Text': row_j[text_col],
                                'Language B': lang_j,
                                'Similarity': round(score * 100, 1),
                                'Cross-Language': lang_i != lang_j,
                                'Target_Column': text_col 
                            })

            # --- FALLBACK LOGIC ---
            else:
                if score >= base_threshold:
                    row_i, row_j = df.iloc[i], df.iloc[j]
                    lang_i = str(row_i.get('language', '')).strip()
                    lang_j = str(row_j.get('language', '')).strip()
                    
                    dynamic_threshold = 0.92 if lang_i == lang_j else 0.85
                    if score >= dynamic_threshold:
                        pair_key = tuple(sorted((i, int(j))))
                        if pair_key not in seen:
                            seen.add(pair_key)
                            pairs.append({
                                'Record A ID': row_i.get('id', i),
                                'Record A Text': row_i[text_col],
                                'Language A': lang_i,
                                'Record B ID': row_j.get('id', j),
                                'Record B Text': row_j[text_col],
                                'Language B': lang_j,
                                'Similarity': round(score * 100, 1),
                                'Cross-Language': lang_i != lang_j,
                                'Target_Column': text_col 
                            })

    result_df = pd.DataFrame(pairs)
    if not result_df.empty:
        result_df = result_df.sort_values('Similarity', ascending=False).reset_index(drop=True)
    return result_df

# ---------------------- Merge Logic ----------------------
def generate_new_record(text_a, lang_a, text_b, lang_b):
    primary_text = text_a if lang_a.lower() == 'english' else text_b
    secondary_text = text_b if lang_a.lower() == 'english' else text_a
    secondary_lang = lang_b if lang_a.lower() == 'english' else lang_a
    
    return {
        "New_ID": f"ITEM_{abs(hash(primary_text)) % 10000}",
        "Primary_Name": primary_text,
        "Primary_Language": "English",
        "Regional_Aliases": {secondary_lang: secondary_text},
        "Resolution_Status": "Merged Confirmed "
    }

# ---------------------- Action Handler ----------------------
def run_action(action: str, text: str, df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if df is None: return [{"type": "text", "content": "No dataset loaded. Please upload a CSV or Excel file first."}]
    if cols is None: cols = extract_column_names(text, df)

    try: 
        if action == "duplicate":
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            blocked_cols = ['id', 'language', 'group_id', 'group', 'unnamed: 0', 'user_id', 'created_at']
            text_cols = [c for c in text_cols if c.lower() not in blocked_cols]
            
            if not text_cols: return [{"type": "text", "content": "No valid text columns found."}]
            
            target_col = [c for c in cols if c in text_cols]
            target_col = target_col[0] if target_col else text_cols[0]

            model, model_type = load_model()
            dup_df = detect_duplicates(df, target_col, 0.85, model, model_type)
            
            if dup_df.empty:
                return [{"type": "text", "content": f" No semantic duplicates found in column `{target_col}`."}]
            else:
                return [
                    {"type": "text", "content": f"⚠️ Found **{len(dup_df)}** duplicate pairs. Sending to Resolution Dashboard..."},
                    {"type": "resolution_dashboard", "content": dup_df}
                ]
                
        if action == "download":
            if st.session_state.get('has_merged', False): return [{"type": "download", "content": df}]
            else: return [{"type": "text", "content": "No changes have been made to the original file. Merge duplicates first!"}]

        if action == "hello": 
            return [{"type": "text", "content": "Heyy!! How can I help you today? Upload a dataset and click the buttons to start the action!"}]
        
        # --- RESTORED ACTIONS ---
        if action in ("structure", "rows"): return [{"type": "text", "content": f"The dataset has **{df.shape[0]} rows**."}]
        if action in ("structure", "columns"): return [{"type": "text", "content": f"The dataset has **{df.shape[1]} columns**."}]
        if action == "describe": return [{"type": "table", "content": df.describe(include='all').T}]
        if action == "head": return [{"type": "table", "content": df.head(5)}]
        if action == "tail": return [{"type": "table", "content": df.tail(5)}]
        
        if action == "feature_types":
            numerical = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
            result_df = pd.DataFrame({
                "Feature Type": ["Numerical", "Categorical"],
                "Columns": [", ".join(numerical) if numerical else "None",
                            ", ".join(categorical) if categorical else "None"]
            })
            return [{"type": "table", "content": result_df}]

        if action == "data_quality":
            missing = df.isnull().sum()[df.isnull().sum() > 0]
            dup_count = int(df.duplicated().sum())
            numeric = df.select_dtypes(include=[np.number])
            outlier_info = {}
            for c in numeric.columns:
                if numeric[c].std(ddof=0) != 0 and not numeric[c].isna().all():
                    z = np.abs((numeric[c] - numeric[c].mean()) / numeric[c].std(ddof=0))
                    if (z > 3).sum() > 0: outlier_info[c] = int((z > 3).sum())

            if missing.empty and dup_count == 0 and not outlier_info:
                return [{"type": "text", "content": " Data Quality is excellent. No missing values, duplicates, or outliers found."}]
            else:
                return [{"type": "data_quality", "content": {"missing": missing.to_dict(), "duplicates": dup_count, "outliers": outlier_info}}]
                
        if action == "insights":
            num_rows, num_cols = df.shape
            missing_cells = df.isnull().sum().sum()
            dup_rows = df.duplicated().sum()
            insight_text = f"** Key Insights Overview:**\n\n* **Size:** The database contains {num_rows} records and {num_cols} attributes.\n* **Completeness:** There are {missing_cells} missing data points in total.\n* **Exact Duplicates:** Found {dup_rows} exact (non-semantic) duplicate rows.\n* **Actionable Next Step:** Use the 'Find duplicates' tool to run a semantic AI scan on the text columns."
            return [{"type": "text", "content": insight_text}]
            
        if action in ("histogram", "bar", "line", "scatter", "heatmap", "pie"):
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]

            if action == "heatmap":
                if len(numeric_cols) < 2: return [{"type": "text", "content": "Not enough numeric columns for a correlation heatmap."}]
                fig = px.imshow(df[numeric_cols].corr(), text_auto=".2f", title="Correlation Heatmap")
                return [{"type": "plotly", "content": fig}]

            elif action == "pie":
                cat_cols = df.select_dtypes(exclude=[np.number]).columns
                blocked_cols = ['id', 'text', 'description', 'name']
                cat_cols = [c for c in cat_cols if c.lower() not in blocked_cols]
                
                if len(cat_cols) > 0:
                    counts = df[cat_cols[0]].value_counts().reset_index()
                    counts.columns = [cat_cols[0], "count"]
                    fig = px.pie(counts, names=cat_cols[0], values="count", title=f"Pie Chart: {cat_cols[0]}")
                    return [{"type": "plotly", "content": fig}]
                else:
                    return [{"type": "text", "content": "No categorical columns available for a pie chart."}]

    except Exception as e:
        return [{"type": "text", "content": f"Sorry — I encountered an error: {str(e)}"}]
    
    return [{"type": "text", "content": "I processed your request, but there's no visual output for this specific command."}]

def run_actions(actions: List[str], text: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    all_results = []
    for action in actions:
        try: all_results.extend(run_action(action, text, df))
        except Exception as e: all_results.append({"type": "text", "content": f"Error in {action}: {e}"})
    return all_results

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title='PAT', layout='wide')

if "remaining_queries" not in st.session_state: st.session_state["remaining_queries"] = [1, 2, 3]
if "max_query_idx" not in st.session_state: st.session_state["max_query_idx"] = 3

predefined_queries = {
    1: "Summary of the dataset", 2: "Want a Data Quality Report?", 3: "what are the feature types?",
    4: "Find duplicates", 5: "Download dataset", 6: "Want a Pie chart?",
    7: "Explore Correlation Heatmap?", 8: "Lets analyze the Key Insights", 9: "Heyy!"
}

if not st.session_state["chat_started"]:
    dark_app_bg = f"background-image: url('data:image/png;base64,{dark_bg}') !important; background-size: cover !important; background-position: center center !important; background-attachment: fixed !important;"
    light_app_bg = f"background-image: url('data:image/png;base64,{light_bg}') !important; background-size: cover !important; background-position: center center !important; background-attachment: fixed !important;"
    overlay_display = "block"
else:
    dark_app_bg = "background-image: none !important; background-color: #0b1120 !important;"
    light_app_bg = "background-image: none !important; background-color: #f8fafc !important;"
    overlay_display = "none"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
.stApp {{ font-family: 'Inter', sans-serif !important; {dark_app_bg} }}
.stApp::before {{ content: ''; position: fixed; inset: 0; background: rgba(10, 18, 35, 0.45); z-index: 0; pointer-events: none; display: {overlay_display} !important; }}
.stApp > * {{ position: relative; z-index: 1; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: transparent !important; color: white !important; }}
.main .block-container {{ max-width: 840px !important; margin: 0 auto !important; padding-bottom: 220px !important; }}
[data-testid="stSidebar"] {{ background: rgba(15, 22, 40, 0.82) !important; backdrop-filter: blur(18px) !important; border-right: 1px solid rgba(255,255,255,0.10) !important; }}
[data-testid="stSidebar"] * {{ color: white !important; }}
.stMarkdown, .stText, p, h1, h2, h3, label, span {{ color: white !important; }}
[data-testid="stFileUploader"] button {{ background-color: #e11d48 !important; color: white !important; border: none !important; padding: 10px 20px !important; border-radius: 20px !important; }}
@media (prefers-color-scheme: light) {{
    .stApp {{ {light_app_bg} }}
    .stApp::before {{ background: rgba(255, 255, 255, 0.25) !important; }}
    .stMarkdown, .stText, p, h1, h2, h3, label, span {{ color: #0f172a !important; }}
    [data-testid="stSidebar"] {{ background: rgba(255, 255, 255, 0.85) !important; border-right: 1px solid rgba(0,0,0,0.10) !important; }}
    [data-testid="stSidebar"] * {{ color: #0f172a !important; }}
}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header('PAT.ai')
    st.header(' Upload Database')
    uploaded = st.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'])
    
    if uploaded is not None:
        if st.session_state['uploaded_filename'] != uploaded.name:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
                st.session_state['df'] = df
                st.session_state['has_merged'] = False
                st.session_state['uploaded_filename'] = uploaded.name
                st.session_state['chat_history'] = [] 
                st.session_state["chat_started"] = False
                st.success(f'Loaded {uploaded.name} — {df.shape[0]} rows, {df.shape[1]} cols')
            except Exception as e:
                st.error(f'Could not load file: {e}')
        else:
            st.success(f'Loaded {uploaded.name} — {st.session_state["df"].shape[0]} rows, {st.session_state["df"].shape[1]} cols')
        
    st.markdown("---")
    st.caption('Cognitive Coders')

# ---------------------- LANDING PAGE ----------------------
if not st.session_state["chat_started"]:
    st.markdown("""
    <style>
    .landing-wrapper { position: fixed; top: 34%; left: 50%; transform: translate(-50%, -50%); width: 100%; text-align: center; z-index: 1000; pointer-events: none; }
    .landing-greeting, .landing-sub { font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 2.2rem; font-weight: 600; color: #ffffff !important; line-height: 1.3; margin: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.4); }
    .landing-sub { margin-top: 0.5rem; font-size: 1.2rem; color: #d1d5db !important;}
    [data-testid="stBottom"] { position: fixed !important; top: 62% !important; left: 50% !important; transform: translate(-50%, -50%) !important; width: 100% !important; max-width: 760px !important; background: transparent !important; padding: 0 !important; z-index: 1000; }
    [data-testid="stBottom"] > div { background: transparent !important; }
    div[data-testid="stChatInput"] { background: rgba(26, 31, 46, 0.95) !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; border-radius: 24px !important; min-height: 130px !important; width: 100% !important; box-shadow: 0 12px 40px rgba(0,0,0,0.4) !important; position: relative !important; padding: 0 !important; }
    div[data-testid="stChatInput"]:focus-within { border: 1px solid rgba(255, 255, 255, 0.2) !important; box-shadow: 0 12px 40px rgba(0,0,0,0.5) !important; }
    div[data-testid="stChatInput"] > div { background: transparent !important; border: none !important; padding: 16px 20px !important; height: 130px !important; width: 100% !important; position: static !important; display: block !important; }
    div[data-testid="stChatInput"] div[data-baseweb="textarea"], div[data-testid="stChatInput"] div[data-baseweb="textarea"] > div { background: transparent !important; border: none !important; }
    div[data-testid="stChatInput"] textarea { font-size: 16px !important; color: white !important; padding: 0 !important; min-height: 60px !important; background: transparent !important; line-height: 1.5 !important; }
    button[data-testid="stChatSendButton"] { position: absolute !important; bottom: 16px !important; right: 16px !important; background: transparent !important; color: white !important; z-index: 10; }
    div[data-testid="stChatInput"]::before { content: "✨ v1.0"; position: absolute; bottom: 16px; left: 20px; background: rgba(255,255,255,0.06); padding: 6px 14px; border-radius: 12px; font-size: 13px; font-weight: 500; color: #d1d5db; pointer-events: none; z-index: 10; }
    [data-testid="stHorizontalBlock"] { position: fixed !important; top: 66% !important; left: 50% !important; transform: translateX(-50%) !important; display: flex !important; justify-content: center !important; flex-wrap: wrap !important; width: 100% !important; max-width: 900px !important; gap: 10px !important; z-index: 1005 !important; }
    [data-testid="stHorizontalBlock"] > div { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
    div[data-testid="stHorizontalBlock"] button { background: rgba(36, 42, 60, 0.7) !important; border: none !important; border-radius: 999px !important; padding: 10px 20px !important; font-size: 13.5px !important; font-weight: 500 !important; color: #e5e7eb !important; box-shadow: none !important; transition: background 0.2s ease !important; }
    div[data-testid="stHorizontalBlock"] button:hover { background: rgba(255, 255, 255, 0.15) !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='landing-wrapper'>
            <div class='landing-greeting'>Good Day, mate</div>
            <div class='landing-sub'>What can I help you with today?</div>
        </div>
    """, unsafe_allow_html=True)

    valid_remaining = [q for q in st.session_state["remaining_queries"] if q in predefined_queries]
    st.session_state["remaining_queries"] = valid_remaining

    cols = st.columns(len(st.session_state["remaining_queries"]))
    for i, q in enumerate(st.session_state["remaining_queries"]):
        label = predefined_queries.get(q)
        if cols[i].button(label, key=f"predef-landing-{q}"):
            st.session_state["chat_started"] = True
            st.session_state["chat_history"].append({"role": "user", "content": label})
            results = run_actions(detect_actions(label), label, st.session_state["df"])
            for result in results:
                st.session_state["chat_history"].append({"role": "assistant", "type": result["type"], "content": result["content"]})
            st.session_state["remaining_queries"].remove(q)
            next_q = st.session_state["max_query_idx"] + 1
            if next_q in predefined_queries:
                st.session_state["remaining_queries"].append(next_q)
                st.session_state["max_query_idx"] = next_q
            st.rerun()

# ---------------------- ACTIVE CHAT ----------------------
else:
    st.markdown("""
    <style>
    [data-testid="stBottom"] { background: transparent !important; }
    [data-testid="stBottom"] > div { max-width: 840px !important; margin: 0 auto !important; background: transparent !important; padding-bottom: 24px !important; }
    div[data-testid="stChatInput"] { background: rgba(30, 36, 51, 0.95) !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; border-radius: 24px !important; min-height: 100px !important; width: 100% !important; box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important; position: relative !important; padding: 0 !important; }
    div[data-testid="stChatInput"]:focus-within { border: 1px solid rgba(255, 255, 255, 0.2) !important; }
    div[data-testid="stChatInput"] > div { background: transparent !important; border: none !important; padding: 16px 20px !important; height: 100px !important; width: 100% !important; position: static !important; display: block !important; }
    div[data-testid="stChatInput"] div[data-baseweb="textarea"], div[data-testid="stChatInput"] div[data-baseweb="textarea"] > div { background: transparent !important; border: none !important; }
    div[data-testid="stChatInput"] textarea { font-size: 16px !important; color: white !important; padding: 0 !important; min-height: 40px !important; background: transparent !important; line-height: 1.5 !important; }
    button[data-testid="stChatSendButton"] { position: absolute !important; bottom: 12px !important; right: 16px !important; background: transparent !important; color: white !important; z-index: 10; }
    div[data-testid="stChatInput"]::before { content: "✨ v1.0"; position: absolute; bottom: 12px; left: 20px; background: rgba(255,255,255,0.06); padding: 6px 14px; border-radius: 12px; font-size: 13px; font-weight: 500; color: #d1d5db; pointer-events: none; z-index: 10; }
    .stChatMessage { background: transparent !important; border: none !important; box-shadow: none !important; padding: 12px 0 !important; backdrop-filter: none !important; }
    .stChatMessage[data-testid="stChatMessage-assistant"] { background: transparent !important; }
    .stChatMessage[data-testid="stChatMessage-user"] { background: rgba(255, 255, 255, 0.05) !important; border-radius: 16px !important; padding: 12px 20px !important; width: fit-content !important; max-width: 80% !important; margin-left: auto !important; flex-direction: row-reverse !important; }
    .stChatMessage[data-testid="stChatMessage-user"] [data-testid="stChatMessageAvatarUser"] { margin-left: 1rem; margin-right: 0; }
    
    /* Active Chat Chips Styling */
    [data-testid="stHorizontalBlock"] { display: flex !important; flex-direction: row !important; justify-content: center !important; flex-wrap: wrap !important; width: 100% !important; gap: 8px !important; margin-top: 24px !important; }
    [data-testid="stHorizontalBlock"] > div { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
    div[data-testid="stHorizontalBlock"] button { background: rgba(36, 42, 60, 0.7) !important; border: none !important; border-radius: 999px !important; padding: 8px 16px !important; font-size: 13px !important; font-weight: 500 !important; color: #e5e7eb !important; transition: background 0.2s ease !important; }
    div[data-testid="stHorizontalBlock"] button:hover { background: rgba(255, 255, 255, 0.15) !important; }
    </style>
    """, unsafe_allow_html=True)

    # Render Chat Loop
    for i, msg in enumerate(st.session_state['chat_history']):
        if msg['role'] == 'user':
            st.chat_message('user').write(msg['content'])
        else:
            with st.chat_message('assistant'):
                if msg['type'] == 'text':
                    st.markdown(msg['content'])
                elif msg['type'] == 'table':
                    st.dataframe(msg['content'])
                elif msg['type'] == 'plotly':
                    st.plotly_chart(msg['content'], use_container_width=True, key=f"plotly_{i}")
                
                # --- CHAT-BASED DOWNLOAD BUTTON ---
                elif msg['type'] == 'download':
                    st.markdown("###  Export Cleaned Master Data")
                    st.caption("Here is your updated dataset.")
                    csv_data = st.session_state['df'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=" Download Cleaned CSV",
                        data=csv_data,
                        file_name="cleaned_master_database.csv",
                        mime="text/csv",
                        type="primary",
                        key=f"dl_btn_{i}" 
                    )
                
                # --- DATA QUALITY REPORT ---
                elif msg['type'] == 'data_quality':
                    issues = msg['content']
                    st.markdown("###  Data Quality Report")
                    if issues.get("missing"): st.markdown(f"⚠️ Missing values: {issues['missing']}")
                    if issues.get("duplicates", 0) > 0: st.markdown(f"⚠️ Found {issues['duplicates']} identical rows")
                    if issues.get("outliers"): st.markdown(f"⚠️ Outliers detected: {list(issues['outliers'].keys())}")
                
                # --- STATE-AWARE RESOLUTION DASHBOARD ---
                elif msg['type'] == 'resolution_dashboard':
                    duplicate_df = msg['content']
                    st.markdown("### Intelligent Data Resolution Engine")
                    st.caption("Review FAISS semantic duplicates and merge them into a single New Record.")
                    
                    for index, row in duplicate_df.iterrows():
                        # Track which buttons have been clicked
                        merge_key = f"merged_{row['Record A ID']}_{row['Record B ID']}"
                        
                        if merge_key not in st.session_state:
                            with st.expander(f"⚠️ {row['Record A Text']} | {row['Record B Text']} (Match: {row['Similarity']}%)"):
                                col1, col2, col3 = st.columns([2, 2, 1])
                                
                                with col1: st.info(f"**Record A ({row['Language A']})**\n\nID: {row['Record A ID']}\n\n{row['Record A Text']}")
                                with col2: st.warning(f"**Record B ({row['Language B']})**\n\nID: {row['Record B ID']}\n\n{row['Record B Text']}")
                                
                                with col3:
                                    st.markdown("**Action:**")
                                    if st.button("✨ Auto-Merge", key=f"btn_{i}_{index}"):
                                        with st.spinner("Merging..."):
                                            new_json = generate_new_record(
                                                row['Record A Text'], row['Language A'],
                                                row['Record B Text'], row['Language B']
                                            )
                                            
                                            current_df = st.session_state['df']
                                            id_a, id_b = row['Record A ID'], row['Record B ID']
                                            
                                            if 'id' in current_df.columns:
                                                current_df = current_df[~current_df['id'].isin([id_a, id_b])]
                                            
                                            new_row = pd.Series(index=current_df.columns, dtype=object)
                                            if 'id' in new_row: new_row['id'] = new_json['New_ID']
                                            if 'language' in new_row: new_row['language'] = 'Multilingual (Merged)'
                                            
                                            target_col = row['Target_Column']
                                            if target_col in new_row: new_row[target_col] = new_json['Primary_Name']
                                            
                                            st.session_state['df'] = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
                                            
                                            st.session_state[merge_key] = True
                                            st.session_state['has_merged'] = True
                                            st.rerun()
                        else:
                            st.success(f"Successfully Merged Records {row['Record A ID']} & {row['Record B ID']} into a New Record!")

    # --- SUGGESTION CHIPS ---
    valid_remaining = [q for q in st.session_state["remaining_queries"] if q in predefined_queries]
    st.session_state["remaining_queries"] = valid_remaining

    if len(st.session_state["remaining_queries"]) > 0:
        cols = st.columns(len(st.session_state["remaining_queries"]))
        for i, q in enumerate(st.session_state["remaining_queries"]):
            label = predefined_queries.get(q)
            if cols[i].button(label, key=f"predef-active-{q}"):
                st.session_state["chat_started"] = True
                st.session_state["chat_history"].append({"role": "user", "content": label})
                results = run_actions(detect_actions(label), label, st.session_state["df"])
                for result in results:
                    st.session_state["chat_history"].append({"role": "assistant", "type": result["type"], "content": result["content"]})
                
                st.session_state["remaining_queries"].remove(q)
                next_q = st.session_state["max_query_idx"] + 1
                if next_q in predefined_queries:
                    st.session_state["remaining_queries"].append(next_q)
                    st.session_state["max_query_idx"] = next_q
                st.rerun()

# User Chat Input
user_input = st.chat_input("Ask PAT to analyze...")
if user_input:
    st.session_state["chat_started"] = True
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    results = run_actions(detect_actions(user_input), user_input, st.session_state["df"])
    for result in results:
        st.session_state["chat_history"].append({"role": "assistant", "type": result["type"], "content": result["content"]})
    st.rerun()