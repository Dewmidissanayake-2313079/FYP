import numpy as np
import pandas as pd
import torch
import clip
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score
)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from collections import defaultdict
import gc
import warnings
warnings.filterwarnings('ignore')


def hamming_loss_multiclass(y_true, y_pred):
    return np.mean(y_true != y_pred)


DATASET_PATH  = 'E:/4 year/IRP/FYP/Datasets/dataset_with_age_survey_based.csv'
METADATA_PATH = 'E:/4 year/IRP/FYP/features/item_metadata.csv'
CLIP_FEAT_PATH = 'E:/4 year/IRP/FYP/features/clip_features.npy'


# Demographic dataset initialization
demo = pd.read_csv(DATASET_PATH)

if 'clothing_type' not in demo.columns:
    def derive_clothing_type(row):
        if pd.notna(row['full_body']):
            if row['full_body'] == 'dress':                     return 'dress'
            if row['full_body'] in ['play_suit', 'jump_suit']:  return 'rompers'
        lb = row['lower_body']
        if pd.notna(lb):
            if lb == 'leggings':  return 'leggings'
            if lb == 'skirt':     return 'skirt'
            if lb in ['pants/trousers', 'jeans', 'shorts', 'athletic_pants']:
                return 'pants'
        return 'top'
    demo['clothing_type'] = demo.apply(derive_clothing_type, axis=1)

DEMO_TO_IMAGE_CAT = {
    'top'            : 'top',
    'pants'          : 'pants',
    'skirt'          : 'skirt',
    'dress'          : 'dress',
    'leggings'       : 'leggings',
    'rompers'        : 'rompers',
    'jacket'         : 'outer',
    'blazer'         : 'outer',
    'coat'           : 'outer',
    'cardigan'       : 'outer',
    'vest_waistcoat' : 'outer',
}

TARGET_COLS = ['clothing_type', 'colour_top', 'colour_bottom',
               'neckline', 'sleeve_length']

demo['colour_bottom'] = demo['colour_bottom'].fillna('unknown')

target_encoders = {}
for col in TARGET_COLS:
    le = LabelEncoder()
    demo[f'{col}_enc'] = le.fit_transform(demo[col])
    target_encoders[col] = le
    print(f"  {col:20s} classes: {list(le.classes_)}")

demo_encoded = pd.get_dummies(demo, columns=['gender', 'occasion'], drop_first=False)

input_cols = (
    ['age'] +
    [c for c in demo_encoded.columns if c.startswith('gender_')] +
    [c for c in demo_encoded.columns if c.startswith('occasion_')]
)

X = demo_encoded[input_cols].values.astype(float)
Y = demo[[f'{c}_enc' for c in TARGET_COLS]].values

print(f"\nInput  shape : {X.shape}")
print(f"Output shape : {Y.shape}")


#  Train / Test split 
train_indices, test_indices = train_test_split(
    range(len(demo)), test_size=0.2, random_state=42
)

X_train, X_test = X[train_indices], X[test_indices]
Y_train, Y_test = Y[train_indices], Y[test_indices]

test_demo_rows = demo.iloc[test_indices].reset_index(drop=True)

print(f"\nTrain : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")


#  Random Forest model initialization
print("\n" + "="*65)
print("  MODEL 1 — RANDOM FOREST (Multi-Output Classifier)")
print("="*65)

rf_base  = RandomForestClassifier(
    n_estimators=100, max_depth=15,
    min_samples_split=5, class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf_model = MultiOutputClassifier(rf_base, n_jobs=1)
rf_model.fit(X_train, Y_train)

Y_pred_rf = rf_model.predict(X_test)

print("\nPer-target metrics (Random Forest):")
print(f"{'Target':<22} {'Accuracy':>10} {'F1-W':>10} {'Precision':>10} {'Recall':>10}")
print("-"*65)
all_rf_f1 = []
for i, col in enumerate(TARGET_COLS):
    acc  = accuracy_score(Y_test[:, i], Y_pred_rf[:, i])
    f1   = f1_score(Y_test[:, i], Y_pred_rf[:, i], average='weighted', zero_division=0)
    prec = precision_score(Y_test[:, i], Y_pred_rf[:, i], average='weighted', zero_division=0)
    rec  = recall_score(Y_test[:, i], Y_pred_rf[:, i], average='weighted', zero_division=0)
    all_rf_f1.append(f1)
    print(f"  {col:<20} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f}")

rf_exact = np.mean(np.all(Y_test == Y_pred_rf, axis=1))
rf_hloss = hamming_loss_multiclass(Y_test, Y_pred_rf)
print(f"\n  Exact Match Ratio : {rf_exact:.4f}")
print(f"  Hamming Loss      : {rf_hloss:.4f}")


# XGBoost model initialization
print("\n" + "="*65)
print("  MODEL 2 — XGBOOST (Multi-Output Classifier)")
print("="*65)

xgb_estimators = []
xgb_losses = []

for i in range(Y_train.shape[1]):
    clf = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42,
        n_jobs=-1, verbosity=0
    )
    clf.fit(X_train, Y_train[:, i], eval_set=[(X_train, Y_train[:, i])], verbose=False)
    xgb_estimators.append(clf)
    xgb_losses.append(clf.evals_result()['validation_0']['mlogloss'])

class CustomMultiOutput:
    def __init__(self, estimators):
        self.estimators_ = estimators
    def predict(self, X):
        return np.column_stack([est.predict(X) for est in self.estimators_])

xgb_model = CustomMultiOutput(xgb_estimators)

Y_pred_xgb = xgb_model.predict(X_test)

# XGBoost Training Loss curve
avg_xgb_loss = np.mean(xgb_losses, axis=0)
plt.figure(figsize=(8, 5))
plt.plot(avg_xgb_loss, label='XGBoost Avg Training Loss', color='green')
plt.title('ML Model - XGBoost Training Loss')
plt.xlabel('Trees (Epochs)')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.show()

print("\nPer-target metrics (XGBoost):")
print(f"{'Target':<22} {'Accuracy':>10} {'F1-W':>10} {'Precision':>10} {'Recall':>10}")
print("-"*65)
all_xgb_f1 = []
for i, col in enumerate(TARGET_COLS):
    acc  = accuracy_score(Y_test[:, i], Y_pred_xgb[:, i])
    f1   = f1_score(Y_test[:, i], Y_pred_xgb[:, i], average='weighted', zero_division=0)
    prec = precision_score(Y_test[:, i], Y_pred_xgb[:, i], average='weighted', zero_division=0)
    rec  = recall_score(Y_test[:, i], Y_pred_xgb[:, i], average='weighted', zero_division=0)
    all_xgb_f1.append(f1)
    print(f"  {col:<20} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f}")

xgb_exact = np.mean(np.all(Y_test == Y_pred_xgb, axis=1))
xgb_hloss = hamming_loss_multiclass(Y_test, Y_pred_xgb)
print(f"\n  Exact Match Ratio : {xgb_exact:.4f}")
print(f"  Hamming Loss      : {xgb_hloss:.4f}")


#  Model Comparison (Stage 1) attribute prediction
print(f"\n{'='*65}")
print("  STAGE 1 MODEL COMPARISON — Attribute Prediction")
print(f"{'='*65}")
print(f"  {'Metric':<30} {'Random Forest':>15} {'XGBoost':>15}")
print("  " + "-"*60)

for i, col in enumerate(TARGET_COLS):
    print(f"  F1 — {col:<24} {all_rf_f1[i]:>15.4f} {all_xgb_f1[i]:>15.4f}")

print("  " + "-"*60)
print(f"  {'Avg F1 (all targets)':<30} {np.mean(all_rf_f1):>15.4f} {np.mean(all_xgb_f1):>15.4f}")
print(f"  {'Exact Match Ratio (↑)':<30} {rf_exact:>15.4f} {xgb_exact:>15.4f}")
print(f"  {'Hamming Loss (↓)':<30} {rf_hloss:>15.4f} {xgb_hloss:>15.4f}")

best_model      = rf_model  if np.mean(all_rf_f1) >= np.mean(all_xgb_f1) else xgb_model
best_model_name = 'Random Forest' if np.mean(all_rf_f1) >= np.mean(all_xgb_f1) else 'XGBoost'
print(f"\n  Best attribute predictor: {best_model_name}")


#  Free training arrays before loading CLIP features 
del X_train, Y_train, Y_pred_rf, Y_pred_xgb
gc.collect()


#  Load item metadata and pre-extracted CLIP features 
item_metadata = pd.read_csv(METADATA_PATH)
clip_features = np.load(CLIP_FEAT_PATH)   # (N, 512)

item_metadata['gender'] = item_metadata['image_name'].apply(
    lambda x: 'male' if 'MEN' in x.upper() and 'WOMEN' not in x.upper() else 'female'
)

norms              = np.linalg.norm(clip_features, axis=1, keepdims=True) + 1e-8
clip_features_norm = clip_features / norms
del clip_features, norms
gc.collect()

print(f"\nItem metadata : {item_metadata.shape}")
print(f"CLIP features : {clip_features_norm.shape}")
print(f"Item classes  : {item_metadata['class_name'].unique().tolist()}")


#  Load CLIP model 
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
print(f"\nCLIP model loaded on: {device}")


#  Occasion with the allowed clothing categories 
OCCASION_EXTRA = {
    'Party'  : ['dress', 'skirt', 'rompers', 'top'],
    'Prom'   : ['dress', 'skirt'],
    'Wedding': ['dress', 'skirt'],
    'Office' : ['top', 'pants', 'skirt', 'outer'],
    'Dating' : ['dress', 'skirt', 'top', 'pants', 'rompers'],
    'Travel' : ['top', 'pants', 'outer'],
    'Sports' : ['top', 'pants', 'leggings'],
}
ACCESSORY_CATS = ['footwear', 'bag', 'belt', 'wrist_wearing',
                  'necklace', 'ring', 'socks', 'eyeglass', 'headwear']


def build_user_input_vector(age, gender, occasion):
    gender_cols   = [c for c in demo_encoded.columns if c.startswith('gender_')]
    occasion_cols = [c for c in demo_encoded.columns if c.startswith('occasion_')]
    row = {'age': age}
    for c in gender_cols:
        row[c] = 1.0 if c == f'gender_{gender}' else 0.0
    for c in occasion_cols:
        row[c] = 1.0 if c == f'occasion_{occasion}' else 0.0
    return np.array([[row[c] for c in input_cols]], dtype=float)


def predict_outfit_attributes(age, gender, occasion, model=None):
    """Stage 1 — predict clothing attributes from demographic inputs."""
    if model is None:
        model = best_model
    x_vec      = build_user_input_vector(age, gender, occasion)
    y_pred_enc = model.predict(x_vec)[0]
    return {col: target_encoders[col].classes_[y_pred_enc[i]]
            for i, col in enumerate(TARGET_COLS)}


def build_clip_text_prompt(attributes, gender, occasion):
    """Build a descriptive CLIP text query from predicted attributes."""
    cat    = DEMO_TO_IMAGE_CAT.get(attributes['clothing_type'], attributes['clothing_type'])
    color  = attributes['colour_top']
    neck   = attributes['neckline'].replace('_', ' ')
    sleeve = attributes['sleeve_length'].replace('_', ' ')
    prompt = (f"A photo of a {color} {neck} {sleeve} sleeve {cat} "
              f"for a {gender} {occasion} outfit")
    return prompt, cat


def encode_text_prompt(prompt):
    """Encode text prompt into 512-dim unit-normalised CLIP embedding."""
    with torch.no_grad():
        tokens     = clip.tokenize([prompt]).to(device)
        text_embed = clip_model.encode_text(tokens)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    return text_embed.cpu().numpy().squeeze()


def recommend_outfit(age, gender, occasion, top_k=10, model=None):
    """
    Stage 1 : RF/XGB predicts attribute profile from demographics.
    Stage 2 : Build CLIP text query → cosine similarity ranking over item pool.
    Returns top-k recommended items as a DataFrame.
    """
    if model is None:
        model = best_model

    attrs = predict_outfit_attributes(age, gender, occasion, model)
    prompt, predicted_cat = build_clip_text_prompt(attrs, gender, occasion)
    text_embed = encode_text_prompt(prompt)

    image_cat    = DEMO_TO_IMAGE_CAT.get(attrs['clothing_type'], attrs['clothing_type'])
    allowed_cats = list(set([image_cat] + OCCASION_EXTRA.get(occasion, []) + ACCESSORY_CATS))
    mask = (
        (item_metadata['gender']     == gender) &
        (item_metadata['class_name'].isin(allowed_cats))
    )
    candidate_df   = item_metadata[mask].copy()
    candidate_idxs = candidate_df.index.tolist()

    if len(candidate_df) < top_k:
        mask           = (item_metadata['gender'] == gender)
        candidate_df   = item_metadata[mask].copy()
        candidate_idxs = candidate_df.index.tolist()

    cand_clip_norm = clip_features_norm[candidate_idxs]
    text_sim       = cand_clip_norm @ text_embed

    candidate_df['text_sim']    = text_sim
    candidate_df['final_score'] = text_sim

    ranked       = candidate_df.sort_values('final_score', ascending=False)
    best_per_cat = ranked.drop_duplicates(subset='class_name')

    if len(best_per_cat) >= top_k:
        top_items = best_per_cat.head(top_k)
    else:
        already   = best_per_cat.index
        extras    = ranked.drop(index=already).head(top_k - len(best_per_cat))
        top_items = pd.concat([best_per_cat, extras])

    top_items = (top_items
                 .sort_values('final_score', ascending=False)
                 .reset_index(drop=True))
    top_items.index += 1

    return (top_items[['item_id', 'image_name', 'class_name',
                        'gender', 'text_sim', 'final_score']],
            attrs, prompt)


#  Recommendation quality metrics 

def get_ground_truth_items(gender, true_clothing_type):
    """Items matching the true clothing category and gender = relevant set."""
    true_cat = DEMO_TO_IMAGE_CAT.get(true_clothing_type, true_clothing_type)
    relevant = item_metadata[
        (item_metadata['gender']     == gender) &
        (item_metadata['class_name'] == true_cat)
    ]['item_id'].tolist()
    return set(relevant)


def precision_at_k(recommended_ids, relevant_ids, k):
    hits = sum(1 for i in recommended_ids[:k] if i in relevant_ids)
    return hits / k if k > 0 else 0.0

def recall_at_k(recommended_ids, relevant_ids, k):
    hits = sum(1 for i in recommended_ids[:k] if i in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0

def f1_score_at_k(recommended_ids, relevant_ids, k):
    p = precision_at_k(recommended_ids, relevant_ids, k)
    r = recall_at_k(recommended_ids, relevant_ids, k)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def rmse_at_k(recommended_scores, recommended_ids, relevant_ids, k):
    y_true = np.array([1.0 if i in relevant_ids else 0.0 for i in recommended_ids[:k]])
    y_pred = np.array(recommended_scores[:k])
    return np.sqrt(np.mean((y_true - y_pred)**2)) if k > 0 else 0.0

def mae_at_k(recommended_scores, recommended_ids, relevant_ids, k):
    y_true = np.array([1.0 if i in relevant_ids else 0.0 for i in recommended_ids[:k]])
    y_pred = np.array(recommended_scores[:k])
    return np.mean(np.abs(y_true - y_pred)) if k > 0 else 0.0

def ndcg_at_k(recommended_ids, relevant_ids, k):
    relevance = [1.0 if i in relevant_ids else 0.0 for i in recommended_ids[:k]]
    relevance += [0.0] * (k - len(relevance))
    dcg  = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(top_k=10, models=None, label=None):
    if models is None:
        models = [('Random Forest', rf_model), ('XGBoost', xgb_model)]
    if label is None:
        label = "STAGE 2 EVALUATION — Recommendation Quality"

    sample_idxs = np.arange(len(test_demo_rows))
    results = defaultdict(lambda: defaultdict(list))

    for idx in sample_idxs:
        row      = test_demo_rows.iloc[idx]
        age      = int(row['age'])
        gender   = row['gender']
        occasion = row['occasion']
        true_type = row['clothing_type']

        relevant_ids = get_ground_truth_items(gender, true_type)
        if not relevant_ids:
            continue

        for model_name, model_obj in models:
            try:
                recs, _, _ = recommend_outfit(
                    age=age, gender=gender, occasion=occasion,
                    top_k=top_k, model=model_obj
                )
                rec_ids    = recs['item_id'].tolist()
                rec_scores = recs['final_score'].tolist()

                results[model_name]['precision'].append(precision_at_k(rec_ids, relevant_ids, top_k))
                results[model_name]['recall'].append(recall_at_k(rec_ids, relevant_ids, top_k))
                results[model_name]['f1_score'].append(f1_score_at_k(rec_ids, relevant_ids, top_k))
                results[model_name]['ndcg'].append(ndcg_at_k(rec_ids, relevant_ids, top_k))
                results[model_name]['rmse'].append(rmse_at_k(rec_scores, rec_ids, relevant_ids, top_k))
                results[model_name]['mae'].append(mae_at_k(rec_scores, rec_ids, relevant_ids, top_k))

            except Exception as e:
                print(f"  [Warning] Skipping idx {idx} ({model_name}): {e}")
                continue

    model_names = [m[0] for m in models]
    print(f"\n{'='*65}")
    print(f"  {label} @ K={top_k}")
    print(f"  Sample size : {len(sample_idxs)} users")
    print(f"{'='*65}")
    header = f"  {'Metric':<25}"
    for mn in model_names:
        header += f" {mn:>14}"
    print(header)
    print("  " + "-"*55)

    summary = {}
    metric_labels = {
        'rmse'     : f'RMSE@{top_k}',
        'mae'      : f'MAE@{top_k}',
        'precision': f'Precision@{top_k}',
        'recall'   : f'Recall@{top_k}',
        'f1_score' : f'F1-Score@{top_k}',
        'ndcg'     : f'NDCG@{top_k}'
    }

    for metric, mlabel in metric_labels.items():
        row_str = f"  {mlabel:<25}"
        scores = {}
        for mn in model_names:
            s = np.mean(results[mn][metric]) if results[mn][metric] else 0.0
            scores[mn] = s
            row_str += f" {s:>14.4f}"
        print(row_str)
        summary[metric] = scores

    return summary



#  EVALUATION — Recommendation Quality
rec_summary = evaluate_recommendations(
    top_k=10,
    label="STAGE 2 — Recommendation Quality (WITH AGE)"
)


print(f"\n{'='*80}")
print("  FINAL EVALUATION SUMMARY — ML Model")
print(f"{'='*80}")

print(f"\n  Stage 1 : Attribute Prediction")
print(f"  {'Metric':<28} {'Random Forest':>15} {'XGBoost':>15}")
print("  " + "-"*60)
for lbl, rf_val, xgb_val in [
    ('Avg F1 (weighted)',  np.mean(all_rf_f1), np.mean(all_xgb_f1)),
    ('Exact Match Ratio',  rf_exact,            xgb_exact),
    ('Hamming Loss',       rf_hloss,            xgb_hloss),
]:
    print(f"  {lbl:<28} {rf_val:>15.4f} {xgb_val:>15.4f}")

print(f"\n  Stage 2 : Recommendation Quality (@ K=10)")
print(f"  {'Metric':<28} {'Random Forest':>15} {'XGBoost':>15}")
print("  " + "-"*60)
for metric, lbl in [('rmse',      'RMSE@10'),
                    ('mae',       'MAE@10'),
                    ('precision', 'Precision@10'),
                    ('recall',    'Recall@10'),
                    ('f1_score',  'F1-Score@10'),
                    ('ndcg',      'NDCG@10')]:
    rf_val  = rec_summary[metric]['Random Forest']
    xgb_val = rec_summary[metric]['XGBoost']
    print(f"  {lbl:<28} {rf_val:>15.4f} {xgb_val:>15.4f}")
