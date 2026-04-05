import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

DATASET_PATH   = 'E:/4 year/IRP/FYP/Datasets/dataset_with_age_survey_based.csv'
METADATA_PATH  = 'E:/4 year/IRP/FYP/features/item_metadata.csv'
CLIP_FEAT_PATH = 'E:/4 year/IRP/FYP/features/clip_features.npy'

EMBED_DIM       = 128    # Two-Tower embedding size
HIDDEN_DIM      = 256    # Hidden layer width for both towers
BATCH_SIZE      = 512
EPOCHS_STAGE1   = 40     # Max epochs for attribute-prediction MLP
EPOCHS_TOWER    = 30     # Max epochs for Two-Tower contrastive training
LR              = 3e-4
TEMPERATURE     = 0.07   # InfoNCE temperature
TOP_K           = 10
CLIP_MODEL_NAME = "ViT-B/32"
SCORE_WEIGHT_TT   = 0.20
SCORE_WEIGHT_TEXT = 0.50
PATIENCE_STAGE1   = 10   # Early stopping patience for MLP
PATIENCE_TOWER    = 5    # Early stopping patience for Two-Tower
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {DEVICE}")


#  Data Preprocessing 
demo = pd.read_csv(DATASET_PATH)

# Derive clothing_type column if not already present in the dataset
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

# Align survey clothing types with DeepFashion item categories
DEMO_TO_IMAGE_CAT = {
    'top': 'top', 'pants': 'pants', 'skirt': 'skirt',
    'dress': 'dress', 'leggings': 'leggings', 'rompers': 'rompers',
    'jacket': 'outer', 'blazer': 'outer', 'coat': 'outer',
    'cardigan': 'outer', 'vest_waistcoat': 'outer',
}

# Ensures recommendations include occasion-appropriate garment types
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

# Target outfit attributes to predict
TARGET_COLS = ['clothing_type', 'colour_top', 'colour_bottom',
               'neckline', 'sleeve_length']

demo['colour_bottom'] = demo['colour_bottom'].fillna('unknown')

# Encode each target attribute column using LabelEncoder
target_encoders = {}
num_classes     = {}
for col in TARGET_COLS:
    le = LabelEncoder()
    demo[f'{col}_enc'] = le.fit_transform(demo[col])
    target_encoders[col] = le
    num_classes[col]     = len(le.classes_)
    print(f"  {col:20s} → {num_classes[col]} classes: {list(le.classes_)}")

# One-hot encode gender and occasion columns for model input
demo_encoded = pd.get_dummies(demo, columns=['gender', 'occasion'], drop_first=False)

# Define input feature columns
input_cols = (
    ['age'] +
    [c for c in demo_encoded.columns if c.startswith('gender_')] +
    [c for c in demo_encoded.columns if c.startswith('occasion_')]
)

# Build input feature matrix X and multi-label target matrix Y
X = demo_encoded[input_cols].values.astype(np.float32)
Y = demo[[f'{c}_enc' for c in TARGET_COLS]].values.astype(np.int64)

print(f"\nInput  shape : {X.shape}")
print(f"Output shape : {Y.shape}")


#  Train / Test split, then normalise age from training data only 
train_indices, test_indices = train_test_split(
    range(len(demo)), test_size=0.2, random_state=42
)
X_train, X_test = X[train_indices].copy(), X[test_indices].copy()
Y_train, Y_test = Y[train_indices], Y[test_indices]
test_demo_rows  = demo.iloc[test_indices].reset_index(drop=True)

AGE_MIN = float(X_train[:, 0].min())
AGE_MAX = float(X_train[:, 0].max())
X_train[:, 0] = (X_train[:, 0] - AGE_MIN) / (AGE_MAX - AGE_MIN + 1e-8)
X_test[:, 0]  = (X_test[:, 0]  - AGE_MIN) / (AGE_MAX - AGE_MIN + 1e-8)
print(f"  Age normalised using training stats: min={AGE_MIN:.0f}, max={AGE_MAX:.0f}")

INPUT_DIM = X_train.shape[1]
print(f"\nTrain : {len(X_train)}  |  Test : {len(X_test)}")


# Item metadata & CLIP features 
item_metadata = pd.read_csv(METADATA_PATH)
clip_features = np.load(CLIP_FEAT_PATH).astype(np.float32)   # (N, 512)

item_metadata['gender'] = item_metadata['image_name'].apply(
    lambda x: 'male' if 'MEN' in x.upper() and 'WOMEN' not in x.upper() else 'female'
)

class_le = LabelEncoder()
item_metadata['class_enc'] = class_le.fit_transform(item_metadata['class_name'])
NUM_ITEM_CLASSES = len(class_le.classes_)

norms              = np.linalg.norm(clip_features, axis=1, keepdims=True) + 1e-8
clip_features_norm = clip_features / norms   # (N, 512) unit vectors

print(f"\nItem metadata : {item_metadata.shape}")
print(f"CLIP features : {clip_features_norm.shape}")
print(f"Item classes  : {item_metadata['class_name'].unique().tolist()}")


#  CLIP model for text encoding
_clip_model      = None
_clip_preprocess = None

def _get_clip_model():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        print(f"  Loading CLIP model ({CLIP_MODEL_NAME})...")
        _clip_model, _clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
        _clip_model.eval()
    return _clip_model, _clip_preprocess


# Initialization of multi head MLP for attribute prediction
class DemoDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MultiHeadMLP(nn.Module):
    """
    Shared encoder + one classification head per target attribute.
    Input FC(HIDDEN) -> BN -> ReLU -> Dropout
          FC(HIDDEN//2) -> BN -> ReLU -> Dropout
          [Head_i: FC(num_classes_i)] for each target
    """
    def __init__(self, input_dim, hidden_dim, num_classes_list, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, nc) for nc in num_classes_list
        ])

    def forward(self, x):
        feat = self.encoder(x)
        return [head(feat) for head in self.heads]


def hamming_loss_multiclass(y_true, y_pred):
    return float(np.mean(y_true != y_pred))


def train_stage1(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits_list = model(xb)
        loss = sum(criterion(logits_list[i], yb[:, i])
                   for i in range(len(TARGET_COLS)))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def compute_val_loss(model, X_np, Y_np, criterion):
    """Compute loss on a held-out validation set without gradient updates."""
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
        yb = torch.tensor(Y_np, dtype=torch.long).to(DEVICE)
        logits_list = model(xb)
        loss = sum(criterion(logits_list[i], yb[:, i])
                   for i in range(len(TARGET_COLS)))
    return loss.item()


def evaluate_stage1(model, X_np):
    """Run inference and return predicted class indices (N, num_targets)."""
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
        logits_list = model(xb)
    return np.stack(
        [logits.argmax(dim=1).cpu().numpy() for logits in logits_list],
        axis=1
    )


print("  STAGE 1 — MULTI-HEAD MLP  (Attribute Prediction)")

# Internal 90/10 train/val split to monitor overfitting
train_sub_idx, val_sub_idx = train_test_split(
    range(len(X_train)), test_size=0.1, random_state=42
)
X_tr,  X_val  = X_train[train_sub_idx], X_train[val_sub_idx]
Y_tr,  Y_val  = Y_train[train_sub_idx], Y_train[val_sub_idx]

nc_list = [num_classes[c] for c in TARGET_COLS]

torch.manual_seed(42)
stage1_net = MultiHeadMLP(INPUT_DIM, HIDDEN_DIM, nc_list).to(DEVICE)
criterion  = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(stage1_net.parameters(), lr=LR, weight_decay=1e-4)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer1, T_max=EPOCHS_STAGE1
)

# Drop_last=True prevents a batch of size 1 crashing BatchNorm1d
train_ds = DemoDataset(X_tr, Y_tr)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

stage1_train_loss = []
stage1_val_loss   = []
best_val_loss_s1  = float('inf')
patience_counter1 = 0
best_state_s1     = None

for epoch in range(1, EPOCHS_STAGE1 + 1):
    tr_loss  = train_stage1(stage1_net, train_dl, optimizer1, criterion)
    val_loss = compute_val_loss(stage1_net, X_val, Y_val, criterion)
    stage1_train_loss.append(tr_loss)
    stage1_val_loss.append(val_loss)
    scheduler1.step()

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:>3}/{EPOCHS_STAGE1}  |  train: {tr_loss:.4f}  val: {val_loss:.4f}")

    # Early stopping — save best weights and stop if val stagnates
    if val_loss < best_val_loss_s1:
        best_val_loss_s1  = val_loss
        patience_counter1 = 0
        best_state_s1     = copy.deepcopy(stage1_net.state_dict())
    else:
        patience_counter1 += 1
        if patience_counter1 >= PATIENCE_STAGE1:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE_STAGE1})")
            break

stage1_net.load_state_dict(best_state_s1)
print(f"  Best val loss: {best_val_loss_s1:.4f}")

plt.figure(figsize=(8, 5))
epochs_run = len(stage1_train_loss)
plt.plot(range(1, epochs_run + 1), stage1_train_loss, label='Train Loss')
plt.plot(range(1, epochs_run + 1), stage1_val_loss,   label='Val Loss', linestyle='--')
plt.title('DL Model - Stage 1 (Multi-Head MLP) Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Removed unused Y_np parameter from evaluate_stage1
Y_pred_mlp = evaluate_stage1(stage1_net, X_test)

print("\nPer-target metrics (Multi-Head MLP):")
print(f"{'Target':<22} {'Accuracy':>10} {'F1-W':>10} {'Precision':>10} {'Recall':>10}")
print("-"*65)
all_mlp_f1 = []
for i, col in enumerate(TARGET_COLS):
    acc  = accuracy_score(Y_test[:, i], Y_pred_mlp[:, i])
    f1   = f1_score(Y_test[:, i], Y_pred_mlp[:, i], average='weighted', zero_division=0)
    prec = precision_score(Y_test[:, i], Y_pred_mlp[:, i], average='weighted', zero_division=0)
    rec  = recall_score(Y_test[:, i], Y_pred_mlp[:, i], average='weighted', zero_division=0)
    all_mlp_f1.append(f1)
    print(f"  {col:<20} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f}")

mlp_exact = np.mean(np.all(Y_test == Y_pred_mlp, axis=1))
mlp_hloss = hamming_loss_multiclass(Y_test, Y_pred_mlp)
print(f"\n  Exact Match Ratio : {mlp_exact:.4f}")
print(f"  Hamming Loss      : {mlp_hloss:.4f}")
print(f"  Avg F1 (all)      : {np.mean(all_mlp_f1):.4f}")


# Two tower model initialization
class UserTower(nn.Module):
    """Maps demographic feature vector to a unit-normalised embedding."""
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ItemTower(nn.Module):
    """Maps [CLIP features || class embedding] to a unit-normalised embedding."""
    def __init__(self, clip_dim, num_classes, class_embed_dim,
                 hidden_dim, embed_dim, dropout=0.3):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(clip_dim + class_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, clip_feat, class_idx):
        ce  = self.class_embed(class_idx)
        inp = torch.cat([clip_feat, ce], dim=-1)
        return F.normalize(self.net(inp), dim=-1)


class TwoTowerModel(nn.Module):
    """Wraps both towers for joint training."""
    def __init__(self, input_dim, clip_dim, num_item_classes,
                 hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM,
                 class_embed_dim=32, dropout=0.3):
        super().__init__()
        self.user_tower = UserTower(input_dim, hidden_dim, embed_dim, dropout)
        self.item_tower = ItemTower(clip_dim, num_item_classes,
                                    class_embed_dim, hidden_dim, embed_dim, dropout)
        self.temperature = nn.Parameter(torch.tensor(TEMPERATURE))

    def forward(self, user_feat, item_clip, item_class):
        u_emb = self.user_tower(user_feat)
        i_emb = self.item_tower(item_clip, item_class)
        return u_emb, i_emb

    def info_nce_loss(self, u_emb, i_emb):
        """Symmetric NT-Xent (InfoNCE) loss — diagonal = positive pairs."""
        t      = self.temperature.clamp(0.01, 1.0)
        logits = (u_emb @ i_emb.T) / t
        labels = torch.arange(len(u_emb), device=u_emb.device)
        loss_u = F.cross_entropy(logits,   labels)
        loss_i = F.cross_entropy(logits.T, labels)
        return (loss_u + loss_i) / 2.0


# Build per-epoch pair sampling infrastructure
cat_gender_to_idxs = defaultdict(list)
for idx, row in item_metadata.iterrows():
    cat_gender_to_idxs[(row['gender'], row['class_name'])].append(idx)

rng = np.random.default_rng(42)

def sample_positive_item(gender, clothing_type):
    img_cat = DEMO_TO_IMAGE_CAT.get(clothing_type, clothing_type)
    pool    = cat_gender_to_idxs.get((gender, img_cat), [])
    if not pool:
        pool = [i for i in range(len(item_metadata))
                if item_metadata.iloc[i]['gender'] == gender]
    return rng.choice(pool)

train_demo_rows = demo.iloc[train_indices].reset_index(drop=True)
# Also prepare a validation set of pairs for Stage 2 early stopping
val_demo_rows   = demo.iloc[test_indices].reset_index(drop=True)


class PairDataset(Dataset):
    def __init__(self, user_feats, item_clips, item_classes):
        self.uf = torch.tensor(user_feats,   dtype=torch.float32)
        self.ic = torch.tensor(item_clips,   dtype=torch.float32)
        self.ik = torch.tensor(item_classes, dtype=torch.long)

    def __len__(self):
        return len(self.uf)

    def __getitem__(self, idx):
        return self.uf[idx], self.ic[idx], self.ik[idx]


def make_pairs(demo_rows):
    """Sample one positive item per user row; return clip feats + class indices."""
    pair_idxs   = np.array([
        sample_positive_item(r['gender'], r['clothing_type'])
        for _, r in demo_rows.iterrows()
    ])
    pair_clips   = clip_features[pair_idxs]
    pair_classes = item_metadata.iloc[pair_idxs]['class_enc'].values
    return pair_clips, pair_classes


def contrastive_val_loss(model, demo_rows):
    """One forward pass on freshly-sampled val pairs → InfoNCE loss."""
    pair_clips, pair_classes = make_pairs(demo_rows)
    ds = PairDataset(X_test, pair_clips, pair_classes)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for uf, ic, ik in dl:
            uf, ic, ik = uf.to(DEVICE), ic.to(DEVICE), ik.to(DEVICE)
            u_emb, i_emb = model(uf, ic, ik)
            total += model.info_nce_loss(u_emb, i_emb).item()
    return total / max(len(dl), 1)


print("  STAGE 2 — TWO-TOWER TRAINING  (Contrastive / InfoNCE)")

torch.manual_seed(42)
two_tower = TwoTowerModel(
    input_dim=INPUT_DIM,
    clip_dim=512,
    num_item_classes=NUM_ITEM_CLASSES,
    hidden_dim=HIDDEN_DIM,
    embed_dim=EMBED_DIM,
).to(DEVICE)

optimizer2 = torch.optim.Adam(two_tower.parameters(), lr=LR, weight_decay=1e-4)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer2, T_max=EPOCHS_TOWER
)

stage2_train_loss = []
stage2_val_loss   = []
best_val_loss_s2  = float('inf')
patience_counter2 = 0
best_state_s2     = None

for epoch in range(1, EPOCHS_TOWER + 1):
    # Re-sample positive items each epoch for greater pair diversity
    pair_clips, pair_classes = make_pairs(train_demo_rows)
    pair_ds = PairDataset(X_train, pair_clips, pair_classes)
    pair_dl = DataLoader(pair_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    two_tower.train()
    total_loss = 0.0
    for uf, ic, ik in pair_dl:
        uf, ic, ik = uf.to(DEVICE), ic.to(DEVICE), ik.to(DEVICE)
        optimizer2.zero_grad()
        u_emb, i_emb = two_tower(uf, ic, ik)
        loss = two_tower.info_nce_loss(u_emb, i_emb)
        loss.backward()
        nn.utils.clip_grad_norm_(two_tower.parameters(), max_norm=1.0)
        optimizer2.step()
        total_loss += loss.item()

    epoch_tr_loss  = total_loss / len(pair_dl)
    epoch_val_loss = contrastive_val_loss(two_tower, val_demo_rows)
    stage2_train_loss.append(epoch_tr_loss)
    stage2_val_loss.append(epoch_val_loss)
    scheduler2.step()

    if epoch % 5 == 0:
        print(f"  Epoch {epoch:>3}/{EPOCHS_TOWER}  |  "
              f"train: {epoch_tr_loss:.4f}  val: {epoch_val_loss:.4f}")

    # Early stopping for Two-Tower
    if epoch_val_loss < best_val_loss_s2:
        best_val_loss_s2  = epoch_val_loss
        patience_counter2 = 0
        best_state_s2     = copy.deepcopy(two_tower.state_dict())
    else:
        patience_counter2 += 1
        if patience_counter2 >= PATIENCE_TOWER:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE_TOWER})")
            break

two_tower.load_state_dict(best_state_s2)
print(f"  Best val loss: {best_val_loss_s2:.4f}")

plt.figure(figsize=(8, 5))
epochs_run2 = len(stage2_train_loss)
plt.plot(range(1, epochs_run2 + 1), stage2_train_loss, label='Train Loss', color='orange')
plt.plot(range(1, epochs_run2 + 1), stage2_val_loss,   label='Val Loss',   color='blue', linestyle='--')
plt.title('DL Model - Stage 2 (Two-Tower) Loss')
plt.xlabel('Epochs')
plt.ylabel('Contrastive Loss')
plt.legend()
plt.grid(True)
plt.show()

# Pre-compute item embeddings for the entire catalogue
print("\n  Pre-computing item embeddings for all catalogue items")
two_tower.eval()
all_item_clips   = torch.tensor(clip_features,                     dtype=torch.float32)
all_item_classes = torch.tensor(item_metadata['class_enc'].values, dtype=torch.long)

ITEM_BATCH = 2048
item_embs_list = []
with torch.no_grad():
    for start in range(0, len(all_item_clips), ITEM_BATCH):
        end  = start + ITEM_BATCH
        ic_b = all_item_clips[start:end].to(DEVICE)
        ik_b = all_item_classes[start:end].to(DEVICE)
        item_embs_list.append(two_tower.item_tower(ic_b, ik_b).cpu())

item_embeddings = torch.cat(item_embs_list, dim=0).numpy()   # (N_items, EMBED_DIM)
print(f"  Item embeddings shape: {item_embeddings.shape}")



def build_user_input_vector(age, gender, occasion):
    """
    Constructs a normalised input feature vector from raw user demographics.
    Age is min-max normalised; gender and occasion are one-hot encoded
    to match the format used during model training.
    """
    gender_cols   = [c for c in demo_encoded.columns if c.startswith('gender_')]
    occasion_cols = [c for c in demo_encoded.columns if c.startswith('occasion_')]
    row = {'age': (float(age) - AGE_MIN) / (AGE_MAX - AGE_MIN + 1e-8)}
    for c in gender_cols:
        row[c] = 1.0 if c == f'gender_{gender}' else 0.0
    for c in occasion_cols:
        row[c] = 1.0 if c == f'occasion_{occasion}' else 0.0
    return np.array([[row[c] for c in input_cols]], dtype=np.float32)


def predict_outfit_attributes_mlp(age, gender, occasion):
    """Stage 1 (MLP) — predict clothing attributes from demographics."""
    x_vec = torch.tensor(build_user_input_vector(age, gender, occasion),
                         dtype=torch.float32).to(DEVICE)
    stage1_net.eval()
    with torch.no_grad():
        logits_list = stage1_net(x_vec)
    y_enc = [logits.argmax(dim=1).item() for logits in logits_list]
    return {col: target_encoders[col].classes_[y_enc[i]]
            for i, col in enumerate(TARGET_COLS)}


def get_user_embedding(age, gender, occasion):
    """Run user feature vector through the User Tower → embedding."""
    x_vec = torch.tensor(build_user_input_vector(age, gender, occasion),
                         dtype=torch.float32).to(DEVICE)
    two_tower.eval()
    with torch.no_grad():
        return two_tower.user_tower(x_vec).cpu().numpy().squeeze()


def _build_text_query(attrs: dict) -> str:
    colour = attrs.get('colour_top', '')
    sleeve = attrs.get('sleeve_length', '').replace('_', ' ')
    neck   = attrs.get('neckline', '').replace('_', ' ')
    ctype  = attrs.get('clothing_type', '')
    return f"a {colour} {sleeve} {neck} {ctype}".strip()


def _encode_text_query(text_query: str) -> np.ndarray:
    model, _ = _get_clip_model()
    tokens   = clip.tokenize([text_query]).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_text(tokens).float()
    return F.normalize(feat, dim=-1).cpu().squeeze().numpy()


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def recommend_outfit_two_tower(age, gender, occasion, top_k=TOP_K):
    """
    Stage 1 : MLP predicts attribute profile from demographics.
    Stage 2 : Two-Tower user embedding + CLIP text query -> combined scoring.
    Returns top-k recommended items as a DataFrame.
    """
    attrs       = predict_outfit_attributes_mlp(age, gender, occasion)
    text_query  = _build_text_query(attrs)
    text_feat   = _encode_text_query(text_query)
    u_emb       = get_user_embedding(age, gender, occasion)

    image_cat    = DEMO_TO_IMAGE_CAT.get(attrs['clothing_type'], attrs['clothing_type'])
    allowed_cats = list(set([image_cat] + OCCASION_EXTRA.get(occasion, []) + ACCESSORY_CATS))
    mask = (
        (item_metadata['gender']     == gender) &
        (item_metadata['class_name'].isin(allowed_cats))
    )
    candidate_df   = item_metadata[mask].copy()
    candidate_idxs = candidate_df.index.tolist()

    if len(candidate_df) < top_k:
        mask           = item_metadata['gender'] == gender
        candidate_df   = item_metadata[mask].copy()
        candidate_idxs = candidate_df.index.tolist()

    cand_clip   = clip_features_norm[candidate_idxs]
    tt_scores   = item_embeddings[candidate_idxs] @ u_emb
    text_scores = cand_clip @ text_feat

    w_sum  = SCORE_WEIGHT_TT + SCORE_WEIGHT_TEXT
    scores = ((SCORE_WEIGHT_TT   / w_sum) * _minmax(tt_scores) +
              (SCORE_WEIGHT_TEXT / w_sum) * _minmax(text_scores))

    candidate_df          = candidate_df.copy()
    candidate_df['score'] = scores

    ranked       = candidate_df.sort_values('score', ascending=False)
    best_per_cat = ranked.drop_duplicates(subset='class_name')

    if len(best_per_cat) >= top_k:
        top_items = best_per_cat.head(top_k)
    else:
        already   = best_per_cat.index
        extras    = ranked.drop(index=already).head(top_k - len(best_per_cat))
        top_items = pd.concat([best_per_cat, extras])

    top_items = (top_items
                 .sort_values('score', ascending=False)
                 .reset_index(drop=True))
    top_items.index += 1

    return (top_items[['item_id', 'image_name', 'class_name', 'gender', 'score']], attrs)


# Recommendation quality metrics calculation
def get_ground_truth_items(gender, true_clothing_type):
    true_cat = DEMO_TO_IMAGE_CAT.get(true_clothing_type, true_clothing_type)
    return set(item_metadata[
        (item_metadata['gender']     == gender) &
        (item_metadata['class_name'] == true_cat)
    ]['item_id'].tolist())


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
    dcg   = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg  = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_two_tower(top_k=TOP_K, label=None):
    if label is None:
        label = "STAGE 2 EVALUATION — Two-Tower Retrieval Quality"

    sample_idxs = np.arange(len(test_demo_rows))
    metrics     = defaultdict(list)

    for idx in sample_idxs:
        row          = test_demo_rows.iloc[idx]
        relevant_ids = get_ground_truth_items(row['gender'], row['clothing_type'])
        if not relevant_ids:
            continue
        try:
            recs, _ = recommend_outfit_two_tower(
                age=int(row['age']), gender=row['gender'],
                occasion=row['occasion'], top_k=top_k
            )
            rec_ids    = recs['item_id'].tolist()
            rec_scores = recs['score'].tolist()
            metrics['precision'].append(precision_at_k(rec_ids, relevant_ids, top_k))
            metrics['recall'].append(recall_at_k(rec_ids, relevant_ids, top_k))
            metrics['f1_score'].append(f1_score_at_k(rec_ids, relevant_ids, top_k))
            metrics['ndcg'].append(ndcg_at_k(rec_ids, relevant_ids, top_k))
            metrics['rmse'].append(rmse_at_k(rec_scores, rec_ids, relevant_ids, top_k))
            metrics['mae'].append(mae_at_k(rec_scores, rec_ids, relevant_ids, top_k))
        except Exception as e:
            print(f"  [Warning] Skipping idx {idx}: {e}")

    print(f"\n{'='*65}")
    print(f"  {label} @ K={top_k}")
    print(f"  Sample size : {len(sample_idxs)} users")
    print(f"{'='*65}")
    print(f"  {'Metric':<25} {'Score':>14}")
    print("  " + "-"*42)

    summary = {}
    metric_labels = {
        'rmse'     : f'RMSE@{top_k}',
        'mae'      : f'MAE@{top_k}',
        'precision': f'Precision@{top_k}',
        'recall'   : f'Recall@{top_k}',
        'f1_score' : f'F1-Score@{top_k}',
        'ndcg'     : f'NDCG@{top_k}',
    }
    for metric, mlabel in metric_labels.items():
        score = np.mean(metrics[metric]) if metrics[metric] else 0.0
        print(f"  {mlabel:<25} {score:>14.4f}")
        summary[metric] = score

    return summary


# Evaluate recommendation quality
two_tower_summary = evaluate_two_tower(
    top_k=TOP_K,
    label="STAGE 2 — Two-Tower Retrieval (WITH AGE)"
)


print("  FINAL EVALUATION SUMMARY — DL Model")

print(f"\n  Stage 1 : Attribute Prediction (Multi-Head MLP)")
print(f"  {'Metric':<28} {'Score':>12}")
print("  " + "-"*42)
for lbl, score in [('Avg F1 (weighted)',  np.mean(all_mlp_f1)),
                   ('Exact Match Ratio',   mlp_exact),
                   ('Hamming Loss',        mlp_hloss)]:
    print(f"  {lbl:<28} {score:>12.4f}")

print(f"\n  Stage 2 : Recommendation Quality (@ K={TOP_K})")
print(f"  {'Metric':<28} {'Score':>12}")
print("  " + "-"*42)
for metric, lbl in [('rmse',      f'RMSE@{TOP_K}'),
                    ('mae',       f'MAE@{TOP_K}'),
                    ('precision', f'Precision@{TOP_K}'),
                    ('recall',    f'Recall@{TOP_K}'),
                    ('f1_score',  f'F1-Score@{TOP_K}'),
                    ('ndcg',      f'NDCG@{TOP_K}')]:
    print(f"  {lbl:<28} {two_tower_summary[metric]:>12.4f}")
