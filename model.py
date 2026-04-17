import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from itertools import combinations
import random
import warnings

warnings.filterwarnings('ignore')

# 1. Load Data & Model
print("Loading dataset and LaBSE model...")
df = pd.read_csv('ultra_complex_multilingual_dataset.csv')
model = SentenceTransformer('LaBSE')

# Fill missing values and PRE-PARSE dates for massive speed/memory optimization
df['name'] = df['name'].fillna('')
df['description'] = df['description'].fillna('')
df['category'] = df['category'].fillna('Unknown')
df['region'] = df['region'].fillna('Unknown')
print("Parsing timestamps...")
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# Pre-compute embeddings for speed
print("Computing LaBSE embeddings (this might take a minute)...")
df['text_for_embed'] = df['name'] + " " + df['description']
embeddings = model.encode(df['text_for_embed'].tolist(), normalize_embeddings=True)
df['embedding'] = list(embeddings)

# 2. Generate Pairs (Memory Optimized)
print("Generating Positive and Negative Pairs...")
positive_pairs = []
negative_pairs = []

# Positives (Same group_id)
grouped = df.groupby('record_group_id')
for _, group in grouped:
    if len(group) > 1:
        for idx1, idx2 in combinations(group.index, 2):
            positive_pairs.append((idx1, idx2, 1))

# --- THE FIX: Cap the number of pairs to save RAM ---
MAX_POSITIVES = 5000
if len(positive_pairs) > MAX_POSITIVES:
    positive_pairs = random.sample(positive_pairs, MAX_POSITIVES)
    print(f"Capped positive pairs to {MAX_POSITIVES} to save memory.")

# Negatives (Different group_id)
all_indices = df.index.tolist()
target_negatives = len(positive_pairs) * 2

while len(negative_pairs) < target_negatives: 
    idx1, idx2 = random.sample(all_indices, 2)
    if df.loc[idx1, 'record_group_id'] != df.loc[idx2, 'record_group_id']:
        negative_pairs.append((idx1, idx2, 0))

all_pairs = positive_pairs + negative_pairs
pair_df = pd.DataFrame(all_pairs, columns=['idx1', 'idx2', 'is_duplicate'])

# 3. Feature Engineering
print(f"Engineering Features for {len(pair_df)} pairs... (This will be fast now)")
features = []

for count, row in pair_df.iterrows():
    if count % 2000 == 0 and count > 0:
        print(f"Processed {count}/{len(pair_df)} pairs...")
        
    r1 = df.loc[row['idx1']]
    r2 = df.loc[row['idx2']]
    
    labse_sim = float(np.dot(r1['embedding'], r2['embedding']))
    name_fuzz = fuzz.ratio(str(r1['name']), str(r2['name'])) / 100.0
    
    cat_A = str(r1['category'])
    cat_B = str(r2['category'])
    region_A = str(r1['region'])
    region_B = str(r2['region'])
    
    # Fast datetime math using pre-parsed columns
    try:
        if pd.notnull(r1['created_at']) and pd.notnull(r2['created_at']):
            time_diff_hours = abs((r1['created_at'] - r2['created_at']).total_seconds()) / 3600.0
        else:
            time_diff_hours = 9999.0
    except:
        time_diff_hours = 9999.0
    
    features.append({
        'labse_sim': labse_sim,
        'name_fuzz': name_fuzz,
        'category_A': cat_A,
        'category_B': cat_B,
        'region_A': region_A,
        'region_B': region_B,
        'time_diff_hours': time_diff_hours,
        'is_duplicate': row['is_duplicate']
    })

training_data = pd.DataFrame(features)

# 4. Train CatBoost
print("Training CatBoost Classifier...")
X = training_data.drop('is_duplicate', axis=1)
y = training_data['is_duplicate']

categorical_features_indices = ['category_A', 'category_B', 'region_A', 'region_B']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cb_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=4,
    cat_features=categorical_features_indices,
    eval_metric='Logloss',
    verbose=50
)

cb_model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 5. Evaluate
predictions = cb_model.predict(X_test)
print("\n--- Model Evaluation ---")
print(classification_report(y_test, predictions))

print("\n--- Feature Importance ---")
feature_importances = cb_model.get_feature_importance()
for score, name in sorted(zip(feature_importances, X.columns), reverse=True):
    print(f"{name}: {round(score, 2)}%")

# 6. Save the model
cb_model.save_model("catboost_duplicate_model.cbm")
print("\n✅ Model saved to 'catboost_duplicate_model.cbm'! Ready for Streamlit.")