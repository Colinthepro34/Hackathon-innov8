"""
PAT.ai - Enterprise Data Resolution Engine
Hackathon Edition (PS 3 & PS 1)
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

# ---------------------- Helper: Background Images ----------------------
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
    'hello': ['hello', 'how are you', 'Heyy!', 'hi']
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

# ---------------------- HACKATHON INNOVATION: FAISS Engine ----------------------
def detect_duplicates(df: pd.DataFrame, text_col: str, threshold: float, model, model_type: str) -> pd.DataFrame:
    texts = df[text_col].astype(str).tolist()

    with st.spinner("🧠 Generating multilingual embeddings..."):
        embeddings = get_embeddings(texts, model, model_type, df=df)

    with st.spinner("⚡ Indexing with FAISS for scalable search..."):
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        
        index = faiss.IndexFlatIP(embeddings_np.shape[1]) 
        index.add(embeddings_np)
        
        k = min(10, len(df))
        distances, indices = index.search(embeddings_np, k)

    pairs = []
    seen = set()
    n = len(df)
    
    for i in range(n):
        for rank, j in enumerate(indices[i]):
            if i == j: continue
            score = float(distances[i][rank])
            
            if score >= threshold:
                pair_key = tuple(sorted((i, int(j))))
                if pair_key not in seen:
                    seen.add(pair_key)
                    row_i, row_j = df.iloc[i], df.iloc[j]
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
                        'Target_Column': text_col # Save this for later merging
                    })

    result_df = pd.DataFrame(pairs)
    if not result_df.empty:
        result_df = result_df.sort_values('Similarity', ascending=False).reset_index(drop=True)
    return result_df

# ---------------------- HACKATHON INNOVATION: Golden Record GenAI ----------------------
def generate_golden_record(text_a, lang_a, text_b, lang_b):
    primary_text = text_a if lang_a.lower() == 'english' else text_b
    secondary_text = text_b if lang_a.lower() == 'english' else text_a
    secondary_lang = lang_b if lang_a.lower() == 'english' else lang_a
    
    return {
        "Universal_ID": f"ITEM_{abs(hash(primary_text)) % 10000}",
        "Primary_Name": primary_text,
        "Primary_Language": "English",
        "Regional_Aliases": {secondary_lang: secondary_text},
        "Resolution_Status": "Merged Confirmed ✅"
    }

# ---------------------- Core Action Handler ----------------------
def run_action(action: str, text: str, df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if df is None: return [{"type": "text", "content": "No dataset loaded. Please upload a CSV or Excel file first."}]
    if cols is None: cols = extract_column_names(text, df)

    try: 
        if action == "duplicate":
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            if not text_cols: return [{"type": "text", "content": "No text columns found."}]
            
            target_col = [c for c in cols if c in text_cols]
            target_col = target_col[0] if target_col else text_cols[0]

            model, model_type = load_model()
            
            # TUNED THRESHOLD: Set to 0.92 to fix the "Amazing vs Terrible" sentiment bug!
            dup_df = detect_duplicates(df, target_col, 0.92, model, model_type)
            
            if dup_df.empty:
                return [{"type": "text", "content": f"✅ No semantic duplicates found in column `{target_col}`."}]
            else:
                return [
                    {"type": "text", "content": f"⚠️ Found **{len(dup_df)}** duplicate pairs. Sending to Resolution Dashboard..."},
                    {"type": "resolution_dashboard", "content": dup_df}
                ]

        if action == "hello": return [{"type": "text", "content": "Heyy!! How can I help you today? Upload the dataset and let's start the action!"}]
        if action in ("structure", "rows"): return [{"type": "text", "content": f"The dataset has **{df.shape[0]} rows**."}]
        if action in ("structure", "columns"): return [{"type": "text", "content": f"The dataset has **{df.shape[1]} columns**."}]
        if action == "describe": return [{"type": "table", "content": df.describe(include='all').T}]
        if action == "head": return [{"type": "table", "content": df.head(5)}]
        if action == "tail": return [{"type": "table", "content": df.tail(5)}]

    except Exception as e:
        return [{"type": "text", "content": f"Sorry — I encountered an error: {str(e)}"}]
    
    return [{"type": "text", "content": "I processed your request, but there's no visual output for this specific command."}]

def run_actions(actions: List[str], text: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    all_results = []
    for action in actions:
        try: all_results.extend(run_action(action, text, df))
        except Exception as e: all_results.append({"type": "text", "content": f"Error in {action}: {e}"})
    return all_results

# ---------------------- Streamlit UI Setup ----------------------
st.set_page_config(page_title='PAT.ai OS', layout='wide')

if "remaining_queries" not in st.session_state: st.session_state["remaining_queries"] = [1, 2, 3]
if "max_query_idx" not in st.session_state: st.session_state["max_query_idx"] = 3

predefined_queries = {
    1: "Summary of the dataset", 2: "Want a Data Quality Report?", 3: "what are the feature types?",
    4: "Numeric Distribution of dataset", 5: "The Target Relationships of dataset", 6: "Want a Pie chart?",
    7: "Explore Correlation Heatmap?", 8: "Lets analyze the Key Insights", 9: "Find duplicates", 10: "Heyy!"
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
    st.header('PAT.ai OS')
    st.caption('Enterprise Master Data Management')
    st.header('📂 Upload Database')
    uploaded = st.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
            st.session_state['df'] = df
            st.success(f'Loaded {uploaded.name} — {df.shape[0]} rows, {df.shape[1]} cols')
        except Exception as e:
            st.error(f'Could not load file: {e}')
    st.caption('Hackathon Edition')

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
    div[data-testid="stChatInput"]::before { content: "✨ PAT.ai Daemon Active"; position: absolute; bottom: 16px; left: 20px; background: rgba(255,255,255,0.06); padding: 6px 14px; border-radius: 12px; font-size: 13px; font-weight: 500; color: #d1d5db; pointer-events: none; z-index: 10; }
    [data-testid="stHorizontalBlock"] { position: fixed !important; top: 66% !important; left: 50% !important; transform: translateX(-50%) !important; display: flex !important; justify-content: center !important; flex-wrap: wrap !important; width: 100% !important; max-width: 900px !important; gap: 10px !important; z-index: 1005 !important; }
    [data-testid="stHorizontalBlock"] > div { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
    div[data-testid="stHorizontalBlock"] button { background: rgba(36, 42, 60, 0.7) !important; border: none !important; border-radius: 999px !important; padding: 10px 20px !important; font-size: 13.5px !important; font-weight: 500 !important; color: #e5e7eb !important; box-shadow: none !important; transition: background 0.2s ease !important; }
    div[data-testid="stHorizontalBlock"] button:hover { background: rgba(255, 255, 255, 0.15) !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='landing-wrapper'>
            <div class='landing-greeting'>PAT.ai Operating System</div>
            <div class='landing-sub'>Intelligent Data Resolution & Analytics</div>
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
    div[data-testid="stChatInput"]::before { content: "✨ PAT.ai Daemon Active"; position: absolute; bottom: 12px; left: 20px; background: rgba(255,255,255,0.06); padding: 6px 14px; border-radius: 12px; font-size: 13px; font-weight: 500; color: #d1d5db; pointer-events: none; z-index: 10; }
    .stChatMessage { background: transparent !important; border: none !important; box-shadow: none !important; padding: 12px 0 !important; backdrop-filter: none !important; }
    .stChatMessage[data-testid="stChatMessage-assistant"] { background: transparent !important; }
    .stChatMessage[data-testid="stChatMessage-user"] { background: rgba(255, 255, 255, 0.05) !important; border-radius: 16px !important; padding: 12px 20px !important; width: fit-content !important; max-width: 80% !important; margin-left: auto !important; flex-direction: row-reverse !important; }
    .stChatMessage[data-testid="stChatMessage-user"] [data-testid="stChatMessageAvatarUser"] { margin-left: 1rem; margin-right: 0; }
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
                
                # --- STATE-AWARE RESOLUTION DASHBOARD ---
                elif msg['type'] == 'resolution_dashboard':
                    duplicate_df = msg['content']
                    st.markdown("### 🛠️ Intelligent Data Resolution Engine")
                    st.caption("Review FAISS semantic duplicates and merge them into universal Golden Records.")
                    
                    for index, row in duplicate_df.iterrows():
                        # We use Session State to track which buttons have been clicked so they don't reappear
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
                                            # Generate JSON
                                            golden_json = generate_golden_record(
                                                row['Record A Text'], row['Language A'],
                                                row['Record B Text'], row['Language B']
                                            )
                                            
                                            # DELETE OLD ROWS AND ADD NEW ROW TO SESSION DATAFRAME
                                            current_df = st.session_state['df']
                                            id_a, id_b = row['Record A ID'], row['Record B ID']
                                            
                                            # Drop old
                                            if 'id' in current_df.columns:
                                                current_df = current_df[~current_df['id'].isin([id_a, id_b])]
                                            
                                            # Create new generic row matching existing columns
                                            new_row = pd.Series(index=current_df.columns, dtype=object)
                                            if 'id' in new_row: new_row['id'] = golden_json['Universal_ID']
                                            if 'language' in new_row: new_row['language'] = 'Multilingual (Golden)'
                                            
                                            # Place the merged text in the target text column
                                            target_col = row['Target_Column']
                                            if target_col in new_row: new_row[target_col] = golden_json['Primary_Name']
                                            
                                            # Save back to memory
                                            st.session_state['df'] = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
                                            
                                            # Mark as merged to hide it
                                            st.session_state[merge_key] = True
                                            st.rerun() # Refresh the UI instantly
                        else:
                            st.success(f"✅ Successfully Merged Records {row['Record A ID']} & {row['Record B ID']} into a Golden Record!")

                    # --- THE DOWNLOAD BUTTON ---
                    st.markdown("---")
                    st.markdown("### 💾 Export Cleaned Master Data")
                    csv_data = st.session_state['df'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Cleaned CSV",
                        data=csv_data,
                        file_name="cleaned_master_database.csv",
                        mime="text/csv",
                        type="primary"
                    )

# User Chat Input
user_input = st.chat_input("Ask PAT.ai to analyze or clean data...")
if user_input:
    st.session_state["chat_started"] = True
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    results = run_actions(detect_actions(user_input), user_input, st.session_state["df"])
    for result in results:
        st.session_state["chat_history"].append({"role": "assistant", "type": result["type"], "content": result["content"]})
    st.rerun()