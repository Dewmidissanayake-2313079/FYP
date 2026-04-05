import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import warnings, os, gc
warnings.filterwarnings('ignore')

from torch_geometric.nn import GATv2Conv

DATASET_PATH   = 'E:/4 year/IRP/FYP/Datasets/dataset_with_age_survey_based.csv'
METADATA_PATH  = 'E:/4 year/IRP/FYP/features/item_metadata.csv'
CLIP_FEAT_PATH = 'E:/4 year/IRP/FYP/features/clip_features.npy'
MODEL_PATH     = 'E:/4 year/IRP/FYP/ML models/gat_model_with_age.pt'
OUTPUT_DIR     = 'E:/4 year/IRP/FYP/ML models'

HIDDEN_DIM   = 128
EMBED_DIM    = 64
NUM_LAYERS   = 2
NUM_HEADS    = 4
DROPOUT      = 0.3
ATTN_DROPOUT = 0.1

TARGET_COLS = ['clothing_type', 'colour_top', 'colour_bottom',
               'neckline', 'sleeve_length']

DEMO_TO_IMAGE_CAT = {
    'top': 'top', 'pants': 'pants', 'skirt': 'skirt',
    'dress': 'dress', 'leggings': 'leggings', 'rompers': 'rompers',
    'jacket': 'outer', 'blazer': 'outer', 'coat': 'outer',
    'cardigan': 'outer', 'vest_waistcoat': 'outer',
}

# Homogeneous edge types — eligible for self-loops (FIX 3)
HOMO_EDGE_TYPES = {
    ('outfit', 'similar',  'outfit'),
    ('outfit', 'same_age', 'outfit'),
    ('item',   'cooccurs', 'item'),
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2000  # Process items in batches to avoid OOM


#  MODEL (same as GAT.py) 
class HeteroGATv2(nn.Module):
    def __init__(self, outfit_dim, item_dim, hidden_dim, embed_dim,
                 num_classes_list, num_layers=2, num_heads=4,
                 dropout=0.3, attn_dropout=0.1, dropedge=0.1):
        super().__init__()
        self.dropout = dropout
        self.dropedge = dropedge
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.outfit_proj = nn.Sequential(
            nn.Linear(outfit_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.item_proj = nn.Sequential(
            nn.Linear(item_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())

        self._edge_types = [
            ('outfit', 'wears', 'item'), ('item', 'worn_by', 'outfit'),
            ('outfit', 'similar', 'outfit'), ('outfit', 'same_age', 'outfit'),
            ('item', 'cooccurs', 'item'),
        ]
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_i in range(num_layers):
            is_last = (layer_i == num_layers - 1)
            layer_convs = nn.ModuleDict()
            for (src_t, rel, dst_t) in self._edge_types:
                key = f"{src_t}__{rel}__{dst_t}"
                is_homo = (src_t, rel, dst_t) in HOMO_EDGE_TYPES
                layer_convs[key] = GATv2Conv(
                    in_channels    = hidden_dim,
                    out_channels   = hidden_dim if is_last else self.head_dim,
                    heads          = num_heads,
                    concat         = not is_last,
                    dropout        = attn_dropout,
                    add_self_loops = is_homo,
                    share_weights  = False,
                )
            self.gat_layers.append(layer_convs)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.embed_proj = nn.Linear(hidden_dim, embed_dim)

        # FIX 5: Separate heads for attribute classification vs link prediction
        self.attr_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, nc)
            ) for nc in num_classes_list
        ])
        self.link_proj_outfit = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ELU())
        self.link_proj_item   = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ELU())

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'outfit': self.outfit_proj(x_dict['outfit']),
            'item':   self.item_proj(x_dict['item']),
        }
        for layer_i in range(self.num_layers):
            x_dict = self._gat_layer(x_dict, edge_index_dict, layer_i)
        return {ntype: self.embed_proj(x) for ntype, x in x_dict.items()}

    def _gat_layer(self, x_dict, edge_index_dict, layer_idx):
        layer_convs = self.gat_layers[layer_idx]
        norm = self.norms[layer_idx]
        msg_dict = defaultdict(list)
        for (src_t, rel, dst_t) in self._edge_types:
            key = f"{src_t}__{rel}__{dst_t}"
            ei = edge_index_dict.get((src_t, rel, dst_t))
            if ei is None or key not in layer_convs:
                continue
            out = layer_convs[key]((x_dict[src_t], x_dict[dst_t]), ei)
            msg_dict[dst_t].append(out)
        new_x = {}
        for ntype in x_dict:
            if msg_dict[ntype]:
                agg = torch.stack(msg_dict[ntype], dim=0).mean(dim=0)
                h   = norm(agg + x_dict[ntype])
                h   = F.elu(h)
                h   = F.dropout(h, p=self.dropout, training=self.training)
                new_x[ntype] = h
            else:
                new_x[ntype] = x_dict[ntype]
        return new_x

    def predict_attributes(self, outfit_emb):
        return [head(outfit_emb) for head in self.attr_heads]

    def predict_links(self, outfit_emb, item_emb):
        # FIX 8: Cosine similarity
        o = F.normalize(self.link_proj_outfit(outfit_emb), dim=-1)
        i = F.normalize(self.link_proj_item(item_emb),   dim=-1)
        return (o * i).sum(dim=-1)



#  MAIN 
print(f"Device: {DEVICE}")

print("\n[1/4] Loading data ...")
demo = pd.read_csv(DATASET_PATH)
item_metadata = pd.read_csv(METADATA_PATH)
clip_features = np.load(CLIP_FEAT_PATH).astype(np.float32)

if 'clothing_type' not in demo.columns:
    def derive_clothing_type(row):
        if pd.notna(row['full_body']):
            if row['full_body'] == 'dress': return 'dress'
            if row['full_body'] in ['play_suit', 'jump_suit']: return 'rompers'
        lb = row['lower_body']
        if pd.notna(lb):
            if lb == 'leggings': return 'leggings'
            if lb == 'skirt': return 'skirt'
            if lb in ['pants/trousers', 'jeans', 'shorts', 'athletic_pants']: return 'pants'
        return 'top'
    demo['clothing_type'] = demo.apply(derive_clothing_type, axis=1)

demo['colour_bottom'] = demo['colour_bottom'].fillna('unknown')
item_metadata['gender'] = item_metadata['image_name'].apply(
    lambda x: 'male' if 'MEN' in x.upper() and 'WOMEN' not in x.upper() else 'female')

target_encoders = {}
num_classes = {}
for col in TARGET_COLS:
    le = LabelEncoder()
    demo[f'{col}_enc'] = le.fit_transform(demo[col])
    target_encoders[col] = le
    num_classes[col] = len(le.classes_)

age_group_le = LabelEncoder().fit(demo['age_group'])
gender_le = LabelEncoder().fit(demo['gender'])
occasion_le = LabelEncoder().fit(demo['occasion'])
demo['age_group_enc'] = age_group_le.transform(demo['age_group'])
demo['gender_enc'] = gender_le.transform(demo['gender'])
demo['occasion_enc'] = occasion_le.transform(demo['occasion'])

print(f"  Outfits: {len(demo)}, Items: {len(item_metadata)}, CLIP: {clip_features.shape}")

# Build outfit features
n_ag = len(age_group_le.classes_)
n_g = len(gender_le.classes_)
n_occ = len(occasion_le.classes_)

ag_oh = np.zeros((len(demo), n_ag), dtype=np.float32)
g_oh = np.zeros((len(demo), n_g), dtype=np.float32)
occ_oh = np.zeros((len(demo), n_occ), dtype=np.float32)
for i in range(len(demo)):
    ag_oh[i, demo['age_group_enc'].iloc[i]] = 1.0
    g_oh[i, demo['gender_enc'].iloc[i]] = 1.0
    occ_oh[i, demo['occasion_enc'].iloc[i]] = 1.0

outfit_features = np.concatenate([ag_oh, g_oh, occ_oh], axis=1)
outfit_dim = outfit_features.shape[1]

# Normalize CLIP features
norms = np.linalg.norm(clip_features, axis=1, keepdims=True) + 1e-8
clip_normed = (clip_features / norms).astype(np.float32)

print(f"\n[2/4] Loading model ...")
nc_list = [num_classes[c] for c in TARGET_COLS]
model = HeteroGATv2(
    outfit_dim=outfit_dim, item_dim=512,
    hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM,
    num_classes_list=nc_list, num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS, dropout=DROPOUT, attn_dropout=ATTN_DROPOUT,
).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print("  Model loaded.")

print(f"\n[3/4] Computing embeddings via projection (batched, no graph attention) ...")
gc.collect()

# Outfit embeddings: small (2863 outfits), can do in one shot
with torch.no_grad():
    outfit_tensor = torch.tensor(outfit_features, dtype=torch.float32).to(DEVICE)
    outfit_hidden = model.outfit_proj(outfit_tensor)
    outfit_emb_raw = model.embed_proj(outfit_hidden)
    # FIX 5 & 8: Apply link projection + L2 normalization
    outfit_embeddings = F.normalize(model.link_proj_outfit(outfit_emb_raw), dim=-1).cpu().numpy()
    del outfit_tensor, outfit_hidden, outfit_emb_raw
    gc.collect()

print(f"  Outfit embeddings: {outfit_embeddings.shape}")

# Item embeddings: process in batches to avoid OOM
n_items = len(clip_normed)
item_embeddings = np.zeros((n_items, EMBED_DIM), dtype=np.float32)

for start in range(0, n_items, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_items)
    with torch.no_grad():
        batch = torch.tensor(clip_normed[start:end], dtype=torch.float32).to(DEVICE)
        hidden = model.item_proj(batch)
        emb_raw = model.embed_proj(hidden)
        # FIX 5 & 8: Apply link projection + L2 normalization
        emb = F.normalize(model.link_proj_item(emb_raw), dim=-1).cpu().numpy()
        item_embeddings[start:end] = emb
        del batch, hidden, emb_raw, emb
    gc.collect()
    print(f"  Items {start:>6}-{end:>6} / {n_items} done")

print(f"  Item embeddings:   {item_embeddings.shape}")

# Build wears edges (outfit -> item mapping, needed for demographic scoring)
print(f"\n[4/4] Building wears edges and saving ...")
rng = np.random.default_rng(42)

cat_gender_to_idxs = defaultdict(list)
for idx, row in item_metadata.iterrows():
    cat_gender_to_idxs[(row['gender'], row['class_name'])].append(idx)

o_src, i_dst = [], []
for oidx, row in demo.iterrows():
    gender = row['gender']
    ctype = row['clothing_type']
    img_cat = DEMO_TO_IMAGE_CAT.get(ctype, ctype)
    pool = cat_gender_to_idxs.get((gender, img_cat), [])
    if not pool:
        pool = [i for i in range(len(item_metadata))
                if item_metadata.iloc[i]['gender'] == gender]
    n_sample = min(5, len(pool))
    sampled = rng.choice(pool, size=n_sample, replace=False)
    for iidx in sampled:
        o_src.append(oidx)
        i_dst.append(iidx)

wears_edges = np.stack([o_src, i_dst]).astype(np.int64)

# Save everything
np.save(os.path.join(OUTPUT_DIR, 'gat_outfit_embeddings.npy'), outfit_embeddings)
np.save(os.path.join(OUTPUT_DIR, 'gat_item_embeddings.npy'), item_embeddings)
np.save(os.path.join(OUTPUT_DIR, 'gat_wears_edges.npy'), wears_edges)

print(f"  Saved: gat_outfit_embeddings.npy ({outfit_embeddings.shape})")
print(f"  Saved: gat_item_embeddings.npy ({item_embeddings.shape})")
print(f"  Saved: gat_wears_edges.npy ({wears_edges.shape})")
print("\nDone! The Streamlit app will now load these instead of computing them.")
