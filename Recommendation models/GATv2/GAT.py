import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import defaultdict
import warnings, os, sys
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

try:
    import torch_geometric
except ImportError:
    print("Installing PyTorch Geometric ...")
    os.system(f"{sys.executable} -m pip install torch_geometric pyg_lib "
              f"torch_scatter torch_sparse torch_cluster torch_spline_conv "
              f"-f https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+cpu.html "
              f"--break-system-packages -q")
    import torch_geometric

from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv
import torch_geometric.transforms as T

# Configuration
DATASET_PATH   = 'E:/4 year/IRP/FYP/Datasets/dataset_with_age_survey_based.csv'
METADATA_PATH  = 'E:/4 year/IRP/FYP/features/item_metadata.csv'
CLIP_FEAT_PATH = 'E:/4 year/IRP/FYP/features/clip_features.npy'

HIDDEN_DIM   = 128
EMBED_DIM    = 64
NUM_LAYERS   = 2
NUM_HEADS    = 4 # Attention heads per GATv2 layer
DROPOUT      = 0.3
ATTN_DROPOUT = 0.1
BATCH_SIZE   = 512
EPOCHS       = 60
LR           = 5e-4 # Adam learning rate
TOP_K        = 10
NEG_RATIO    = 10
FOCAL_GAMMA  = 2.0
PATIENCE     = 10
KNN_K        = 5

SAVE_PATH_WITH_AGE = 'gat_model_with_age.pt'
SAVE_PATH_NO_AGE   = 'gat_model_no_age.pt'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Homogeneous edge types eligible for self-loops
HOMO_EDGE_TYPES = {
    ('outfit', 'similar',  'outfit'),
    ('outfit', 'same_age', 'outfit'),
    ('item',   'cooccurs', 'item'),
}

print(f"Device: {DEVICE}  |  PyG version: {torch_geometric.__version__}")


# Data Loading & Preprocessing 
demo = pd.read_csv(DATASET_PATH)
item_metadata = pd.read_csv(METADATA_PATH)
clip_features = np.load(CLIP_FEAT_PATH).astype(np.float32)

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

demo['colour_bottom'] = demo['colour_bottom'].fillna('unknown')

# Infer gender from image filename convention in dataset
item_metadata['gender'] = item_metadata['image_name'].apply(
    lambda x: 'male' if 'MEN' in x.upper() and 'WOMEN' not in x.upper() else 'female'
)

# Label-encode all multi-class prediction targets
TARGET_COLS = ['clothing_type', 'colour_top', 'colour_bottom', 'neckline', 'sleeve_length']

target_encoders = {}
num_classes     = {}
for col in TARGET_COLS:
    le = LabelEncoder()
    demo[f'{col}_enc'] = le.fit_transform(demo[col])
    target_encoders[col] = le
    num_classes[col]     = len(le.classes_)
    print(f"  {col:20s} -> {num_classes[col]} classes")

# Encode demographic features used as outfit node attributes
age_group_le = LabelEncoder().fit(demo['age_group'])
gender_le    = LabelEncoder().fit(demo['gender'])
occasion_le  = LabelEncoder().fit(demo['occasion'])
demo['age_group_enc'] = age_group_le.transform(demo['age_group'])
demo['gender_enc']    = gender_le.transform(demo['gender'])
demo['occasion_enc']  = occasion_le.transform(demo['occasion'])

class_le = LabelEncoder()
item_metadata['class_enc'] = class_le.fit_transform(item_metadata['class_name'])

# Mapping from survey clothing types to DeepFashion category names used in item_metadata
DEMO_TO_IMAGE_CAT = {
    'top': 'top', 'pants': 'pants', 'skirt': 'skirt', 'dress': 'dress',
    'leggings': 'leggings', 'rompers': 'rompers', 'jacket': 'outer',
    'blazer': 'outer', 'coat': 'outer', 'cardigan': 'outer', 'vest_waistcoat': 'outer',
}

# Occasion-specific item categories
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

print(f"\n  Outfits: {len(demo)}  |  Items: {len(item_metadata)}  |  CLIP: {clip_features.shape}")


# Graph Construction 

data = HeteroData()

n_age_groups = len(age_group_le.classes_)
n_genders    = len(gender_le.classes_)
n_occasions  = len(occasion_le.classes_)

# Build one-hot encodings for each demographic dimension
age_group_onehot = np.eye(n_age_groups, dtype=np.float32)[demo['age_group_enc'].values]
gender_onehot    = np.eye(n_genders,    dtype=np.float32)[demo['gender_enc'].values]
occasion_onehot  = np.eye(n_occasions,  dtype=np.float32)[demo['occasion_enc'].values]

# Concat demographic into outfit feature vectors
outfit_features        = np.concatenate([age_group_onehot, gender_onehot, occasion_onehot], axis=1)
outfit_features_no_age = np.concatenate([gender_onehot, occasion_onehot], axis=1)

data['outfit'].x = torch.tensor(outfit_features, dtype=torch.float32)
data['outfit'].y = torch.tensor(
    demo[[f'{c}_enc' for c in TARGET_COLS]].values, dtype=torch.long
)
print(f"  Outfit features (with age): {data['outfit'].x.shape}")

# L2-normalise CLIP features
norms       = np.linalg.norm(clip_features, axis=1, keepdims=True) + 1e-8
clip_normed = (clip_features / norms).astype(np.float32)
data['item'].x = torch.tensor(clip_normed, dtype=torch.float32)
print(f"  Item features             : {data['item'].x.shape}")

# Edges: (outfit, wears, item)
cat_gender_to_idxs = defaultdict(list)
for idx, row in item_metadata.iterrows():
    cat_gender_to_idxs[(row['gender'], row['class_name'])].append(idx)

rng = np.random.default_rng(42)
outfit_src, item_dst = [], []
for oidx, row in demo.iterrows():
    img_cat = DEMO_TO_IMAGE_CAT.get(row['clothing_type'], row['clothing_type'])

    # Retrieve items of the correct gender + clothing category
    pool    = cat_gender_to_idxs.get((row['gender'], img_cat), [])
    if not pool:
        pool = list(item_metadata[item_metadata['gender'] == row['gender']].index)
    for iidx in rng.choice(pool, size=min(5, len(pool)), replace=False):
        outfit_src.append(oidx)
        item_dst.append(iidx)

outfit_src, item_dst = np.array(outfit_src), np.array(item_dst)
data['outfit', 'wears',   'item'].edge_index = torch.tensor(
    np.stack([outfit_src, item_dst]), dtype=torch.long)
data['item',   'worn_by', 'outfit'].edge_index = torch.tensor(
    np.stack([item_dst, outfit_src]), dtype=torch.long)
print(f"  (outfit->item) edges      : {len(outfit_src)}")

# Edges: (outfit, similar, outfit)

# encodes style affinity without explicit visual features
demo['sim_key'] = demo['gender'] + '_' + demo['occasion'] + '_' + demo['age_group']
sim_src, sim_dst = [], []
for key, indices in demo.groupby('sim_key').groups.items():
    indices = list(indices)
    if len(indices) < 2:
        continue
    if len(indices) > 50:
        for i in indices:
            for j in rng.choice(indices, size=min(10, len(indices)-1), replace=False):
                if i != j:
                    sim_src.append(i); sim_dst.append(j)
    else:
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                sim_src += [indices[i], indices[j]]
                sim_dst += [indices[j], indices[i]]
data['outfit', 'similar', 'outfit'].edge_index = torch.tensor(
    np.stack([np.array(sim_src), np.array(sim_dst)]), dtype=torch.long)
print(f"  (outfit<->outfit) edges   : {len(sim_src)}")

# Edges: (outfit, same_age, outfit)

# Removing these edges constitutes the ablation study.
age_src, age_dst = [], []
for key, indices in demo.groupby('age_group').groups.items():
    indices = list(indices)
    if len(indices) < 2:
        continue

    # sample 10 neighbours per outfit within same age group
    for i in indices:
        for j in rng.choice(indices, size=min(10, len(indices)-1), replace=False):
            if i != j:
                age_src.append(i); age_dst.append(j)
data['outfit', 'same_age', 'outfit'].edge_index = torch.tensor(
    np.stack([np.array(age_src), np.array(age_dst)]), dtype=torch.long)
print(f"  (same_age) edges          : {len(age_src)}")

# Edges: (item, cooccurs, item)
co_src, co_dst = [], []
for img, indices in item_metadata.groupby('image_name').groups.items():
    indices = list(indices)
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            co_src += [indices[i], indices[j]]
            co_dst += [indices[j], indices[i]]
data['item', 'cooccurs', 'item'].edge_index = torch.tensor(
    np.stack([np.array(co_src), np.array(co_dst)]), dtype=torch.long)
print(f"  (item<->item) edges       : {len(co_src)}")

data = T.ToUndirected()(data)
print(f"\n  Node types : {data.node_types}")
print(f"  Edge types : {data.edge_types}")


# Train / Val / Test Split 

# Stratified temporal split: 80% train, 10% val, 10% test
train_idx, test_idx = train_test_split(range(len(demo)), test_size=0.2, random_state=42)
train_idx, val_idx  = train_test_split(train_idx,        test_size=0.1, random_state=42)

train_mask = torch.zeros(len(demo), dtype=torch.bool)
val_mask   = torch.zeros(len(demo), dtype=torch.bool)
test_mask  = torch.zeros(len(demo), dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True

data['outfit'].train_mask = train_mask
data['outfit'].val_mask   = val_mask
data['outfit'].test_mask  = test_mask
print(f"  Train: {train_mask.sum().item()}  Val: {val_mask.sum().item()}  Test: {test_mask.sum().item()}")


# Focal Loss 
class FocalLoss(nn.Module):

    """
    Focal Loss for multi-class classification.
    Down-weights easy examples so the model focuses on hard/minority classes.
 
    Args:
        gamma     : Focusing parameter. gamma=0 recovers standard cross-entropy.
        alpha     : Per-class weight tensor (inverse class frequency) to handle
                    class imbalance on top of the focusing term.
        reduction : 'mean' or 'sum' aggregation over the batch.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # clamp prevents (1-p_t)^gamma producing NaN when p_t -> 1
        p_t   = torch.exp(-ce_loss).clamp(min=1e-8, max=1.0 - 1e-8)
        focal = ((1 - p_t) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean() if self.reduction == 'mean' else focal.sum()


def compute_class_weights(labels):
"""
    Compute normalised inverse-frequency class weights.
    Prevents zero-division for unseen classes by flooring counts at 1.
    Returns a float32 tensor of shape (num_classes,).
"""
    counts   = np.maximum(np.bincount(labels), 1)
    inv_freq = 1.0 / counts
    return torch.tensor(inv_freq / inv_freq.sum() * len(counts), dtype=torch.float32)


focal_criteria = {}
for col in TARGET_COLS:
    alpha = compute_class_weights(demo[f'{col}_enc'].values).to(DEVICE)
    focal_criteria[col] = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha)
    print(f"  Focal [{col:20s}] min={alpha.min():.3f}  max={alpha.max():.3f}")


# Model Definition 
class HeteroGATv2(nn.Module):

    def __init__(self, outfit_dim, item_dim, hidden_dim, embed_dim,
                 num_classes_list, num_layers=2, num_heads=4,
                 dropout=0.3, attn_dropout=0.1, dropedge=0.1):
        super().__init__()
        self.dropout    = dropout
        self.dropedge   = dropedge
        self.num_layers = num_layers
        self.num_heads  = num_heads

        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        # Input projections: align heterogeneous feature spaces into hidden_dim
        self.outfit_proj = nn.Sequential(
            nn.Linear(outfit_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.item_proj = nn.Sequential(
            nn.Linear(item_dim,   hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())

        self._edge_types = [
            ('outfit', 'wears',    'item'),
            ('item',   'worn_by',  'outfit'),
            ('outfit', 'similar',  'outfit'),
            ('outfit', 'same_age', 'outfit'),
            ('item',   'cooccurs', 'item'),
        ]

        self.gat_layers = nn.ModuleList()
        self.norms      = nn.ModuleList()

        for layer_i in range(num_layers):
            is_last     = (layer_i == num_layers - 1)
            layer_convs = nn.ModuleDict()
            for (src_t, rel, dst_t) in self._edge_types:
                key     = f"{src_t}__{rel}__{dst_t}"
                is_homo = (src_t, rel, dst_t) in HOMO_EDGE_TYPES
                # FIX 3: self-loops only on homogeneous (same-node-type) edges
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

         # Final projection: hidden_dim -> embed_dim 
        self.embed_proj = nn.Linear(hidden_dim, embed_dim)

        # Separate heads for attribute classification vs link prediction
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
       # Project heterogeneous inputs into shared hidden space
        x_dict = {
            'outfit': self.outfit_proj(x_dict['outfit']),
            'item':   self.item_proj(x_dict['item']),
        }
        # Optionally drop random edges to reduce over-fitting (DropEdge)
        if self.training and self.dropedge > 0:
            edge_index_dict = self._apply_dropedge(edge_index_dict)

        #  Stack GATv2 message-passing layers
        for layer_i in range(self.num_layers):
            x_dict = self._gat_layer(x_dict, edge_index_dict, layer_i)

        # Project to final embedding dimension
        return {ntype: self.embed_proj(x) for ntype, x in x_dict.items()}

    def _gat_layer(self, x_dict, edge_index_dict, layer_idx):

        """
        Single GATv2 layer: run all edge-type convolutions, aggregate messages
        per destination node type, apply residual connection and LayerNorm.
        """
        layer_convs = self.gat_layers[layer_idx]
        norm        = self.norms[layer_idx]
        msg_dict    = defaultdict(list)

        for (src_t, rel, dst_t) in self._edge_types:
            key = f"{src_t}__{rel}__{dst_t}"
            ei  = edge_index_dict.get((src_t, rel, dst_t))
            if ei is None or key not in layer_convs:
                continue
            out = layer_convs[key]((x_dict[src_t], x_dict[dst_t]), ei)
            msg_dict[dst_t].append(out)

        new_x = {}
        for ntype in x_dict:
            if msg_dict[ntype]:
                agg = torch.stack(msg_dict[ntype], dim=0).mean(dim=0)
                h   = norm(agg + x_dict[ntype])    # residual connection
                h   = F.elu(h)
                h   = F.dropout(h, p=self.dropout, training=self.training)
                new_x[ntype] = h
            else:
                new_x[ntype] = x_dict[ntype]
        return new_x

    def _apply_dropedge(self, edge_index_dict):
        """
        DropEdge regularisation: randomly zero out a fraction of edges each
        training step, preventing the model from over-relying on specific edges.
        """
        new_dict = {}
        for etype, ei in edge_index_dict.items():
            keep_mask = torch.rand(ei.shape[1], device=ei.device) > self.dropedge
            new_dict[etype] = ei[:, keep_mask]
        return new_dict

    def predict_attributes(self, outfit_emb): # Run each attribute head on outfit embeddings
        return [head(outfit_emb) for head in self.attr_heads]

    def predict_links(self, outfit_emb, item_emb):
        # Cosine similarity — L2-normalise both sides
        # Dedicated link heads separate from attr_heads
        o = F.normalize(self.link_proj_outfit(outfit_emb), dim=-1)
        i = F.normalize(self.link_proj_item(item_emb),   dim=-1)
        return (o * i).sum(dim=-1)


# Utility Functions 

def get_train_subgraph_edges(edge_index_dict, train_mask, device):
    """
    (Message Passing Data Leak prevention):
    During training, remove any edge whose outfit endpoint is a test node.
    """
    masked = {}
    for etype, ei in edge_index_dict.items():
        src_type, rel, dst_type = etype
        keep = torch.ones(ei.shape[1], dtype=torch.bool, device=device)
        if src_type == 'outfit':
            keep &= train_mask.to(device)[ei[0]]
        if dst_type == 'outfit':
            keep &= train_mask.to(device)[ei[1]]
        masked[etype] = ei[:, keep]
    return masked


def dynamic_link_item(item_feat_norm: np.ndarray,
                      all_item_feats_norm: np.ndarray,
                      k: int = KNN_K) -> np.ndarray:
    """
    Find K nearest catalogue neighbours for a cold item
    using cosine similarity in CLIP space.

    KNN is computed in CLIP space (not learned embedding space) because
    the model has no prior embedding for a truly unseen item.

    Returns: np.ndarray of shape (k,) - neighbour indices (self excluded)
    """
    sims  = all_item_feats_norm @ item_feat_norm
    top_k = np.argsort(sims)[::-1][1:k + 1].copy()  # skip rank-0 (self)
    return top_k


def infer_cold_start_item(model: nn.Module,
                          new_item_feat: np.ndarray,
                          new_item_idx: int,
                          base_data: HeteroData,
                          clip_normed_all: np.ndarray,
                          knn_k: int = KNN_K,
                          device: torch.device = DEVICE) -> np.ndarray:
    """
    For a catalogue item with zero training interactions:
      1. Finds KNN neighbours in CLIP space via dynamic_link_item()
      2. Injects synthetic bidirectional (item, cooccurs, item) edges into
         a temporary deep-copy of the graph (original graph NOT mutated)
      3. Runs a full GATv2 forward pass on the augmented graph
      4. Returns the cold item's embedding from the learned space
    """
    neighbour_idxs = dynamic_link_item(new_item_feat, clip_normed_all, k=knn_k)

    aug_data = copy.deepcopy(base_data)

    new_src = torch.tensor(neighbour_idxs,                       dtype=torch.long)
    new_dst = torch.tensor([new_item_idx] * len(neighbour_idxs), dtype=torch.long)

    old_ei = aug_data['item', 'cooccurs', 'item'].edge_index
    aug_data['item', 'cooccurs', 'item'].edge_index = torch.cat(
        [old_ei,
         torch.stack([new_src, new_dst]),
         torch.stack([new_dst, new_src])],
        dim=1
    )

    model.eval()
    aug_data = aug_data.to(device)
    with torch.no_grad():
        emb_dict = model(aug_data.x_dict, aug_data.edge_index_dict)

    return emb_dict['item'][new_item_idx].cpu().numpy()


def build_outfit_positives(pos_src_np, pos_dst_np):
    """
    BPR Negative Bias:
    Build {outfit_idx: set(positive_item_idxs)} for rejection-loop sampling.
    """
    d = defaultdict(set)
    for o, i in zip(pos_src_np, pos_dst_np):
        d[int(o)].add(int(i))
    return d


def hamming_loss_multiclass(y_true, y_pred):
    return float(np.mean(y_true != y_pred))


# Ranking Metric Helpers 

def precision_at_k(rec, rel, k):
    return sum(1 for i in rec[:k] if i in rel) / k if k > 0 else 0.0

def recall_at_k(rec, rel, k):
    return sum(1 for i in rec[:k] if i in rel) / len(rel) if rel else 0.0

def f1_at_k(rec, rel, k):
    p = precision_at_k(rec, rel, k)
    r = recall_at_k(rec, rel, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def ndcg_at_k(rec, rel, k):
    hits  = [1.0 if i in rel else 0.0 for i in rec[:k]]
    hits += [0.0] * (k - len(hits))
    dcg   = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal = sorted(hits, reverse=True)
    idcg  = sum(h / np.log2(i + 2) for i, h in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def map_at_k(rec, rel, k):
    n_rel, p_sum = 0, 0.0
    for i, r in enumerate(rec[:k]):
        if r in rel:
            n_rel += 1; p_sum += n_rel / (i + 1)
    return p_sum / min(len(rel), k) if rel else 0.0


# model Training 

def train_gat(data, outfit_dim, label="GATv2", epochs=EPOCHS):

"""BPR (Bayesian Personalised Ranking) maximises the margin between
    positive item scores and negative item scores per outfit, teaching
    the model to rank relevant items above irrelevant ones.
 """

    nc_list = [num_classes[c] for c in TARGET_COLS]
    torch.manual_seed(42)
    model = HeteroGATv2(
        outfit_dim=outfit_dim, item_dim=512,
        hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM,
        num_classes_list=nc_list, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, dropout=DROPOUT,
        attn_dropout=ATTN_DROPOUT, dropedge=0.1,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    data_dev       = data.to(DEVICE)
    train_mask_dev = data_dev['outfit'].train_mask
    val_mask_dev   = data_dev['outfit'].val_mask

    pos_src = data['outfit', 'wears', 'item'].edge_index[0]
    pos_dst = data['outfit', 'wears', 'item'].edge_index[1]
    tr_pos_mask   = data['outfit'].train_mask[pos_src]
    train_pos_src = pos_src[tr_pos_mask].to(DEVICE)
    train_pos_dst = pos_dst[tr_pos_mask].to(DEVICE)

    # per-outfit positive sets for rejection-loop negative sampling
    outfit_positives = build_outfit_positives(
        train_pos_src.cpu().numpy(), train_pos_dst.cpu().numpy()
    )

    item_genders       = item_metadata['gender'].values
    male_item_t        = torch.tensor(np.where(item_genders == 'male')[0],   dtype=torch.long, device=DEVICE)
    female_item_t      = torch.tensor(np.where(item_genders == 'female')[0], dtype=torch.long, device=DEVICE)
    train_genders      = demo.iloc[train_pos_src.cpu().numpy()]['gender'].values
    male_mask_rep_base = torch.tensor(train_genders == 'male', dtype=torch.bool, device=DEVICE)

    best_val_loss  = float('inf')
    patience_count = 0
    best_state     = None
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # training-only subgraph — exclude test nodes from message passing
        train_edge_dict = get_train_subgraph_edges(
            data_dev.edge_index_dict, data_dev['outfit'].train_mask, DEVICE
        )
        emb_dict = model(data_dev.x_dict, train_edge_dict)

        # Loss 1: Focal attribute classification 
        out_tr   = emb_dict['outfit'][train_mask_dev]
        y_tr     = data_dev['outfit'].y[train_mask_dev]
        cls_loss = sum(
            focal_criteria[col](model.predict_attributes(out_tr)[i], y_tr[:, i])
            for i, col in enumerate(TARGET_COLS)
        )

        # Loss 2: BPR link prediction 
        pos_o_emb  = emb_dict['outfit'][train_pos_src]
        pos_i_emb  = emb_dict['item'][train_pos_dst]
        pos_scores = model.predict_links(pos_o_emb, pos_i_emb)

        num_pos    = len(train_pos_src)
        total_negs = num_pos * NEG_RATIO
        neg_dst    = torch.zeros(total_negs, dtype=torch.long, device=DEVICE)

        m_mask_rep = male_mask_rep_base.repeat_interleave(NEG_RATIO)
        f_mask_rep = ~m_mask_rep
        if m_mask_rep.any():
            rand_m = torch.randint(0, len(male_item_t), (m_mask_rep.sum(),), device=DEVICE)
            neg_dst[m_mask_rep] = male_item_t[rand_m]
        if f_mask_rep.any():
            rand_f = torch.randint(0, len(female_item_t), (f_mask_rep.sum(),), device=DEVICE)
            neg_dst[f_mask_rep] = female_item_t[rand_f]

        # rejection loop — replace any negative that is a true positive
        # Without this, ~1-5% of "negatives" are actually relevant items, poisoning the loss.
        neg_dst_np  = neg_dst.cpu().numpy()
        pos_src_rep = train_pos_src.cpu().numpy().repeat(NEG_RATIO)
        for k_idx, (o, n) in enumerate(zip(pos_src_rep, neg_dst_np)):
            if n in outfit_positives.get(int(o), set()):
                pool  = male_item_t if train_genders[k_idx // NEG_RATIO] == 'male' else female_item_t
                new_n = pool[torch.randint(0, len(pool), (1,)).item()].item()
                while new_n in outfit_positives.get(int(o), set()):
                    new_n = pool[torch.randint(0, len(pool), (1,)).item()].item()
                neg_dst_np[k_idx] = new_n
        neg_dst = torch.tensor(neg_dst_np, dtype=torch.long, device=DEVICE)

        neg_i_emb  = emb_dict['item'][neg_dst]
        pos_o_rep  = pos_o_emb.repeat_interleave(NEG_RATIO, dim=0)
        neg_scores = model.predict_links(pos_o_rep, neg_i_emb).view(-1, NEG_RATIO)

        # BPR objective
        pos_exp   = pos_scores.unsqueeze(1).expand_as(neg_scores)
        link_loss = -F.logsigmoid(pos_exp - neg_scores).mean()

        # Combined loss classification dominates (weight 1.0), link auxiliary (weight 0.5)
        loss = cls_loss + 0.5 * link_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())

        # Validation of full graph
        model.eval()
        with torch.no_grad():
            emb_val = model(data_dev.x_dict, data_dev.edge_index_dict)
            out_val = emb_val['outfit'][val_mask_dev]
            y_val   = data_dev['outfit'].y[val_mask_dev]
            val_cls = sum(
                focal_criteria[col](model.predict_attributes(out_val)[i], y_val[:, i])
                for i, col in enumerate(TARGET_COLS)
            )
        val_losses.append(val_cls.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  |  "
                  f"focal_cls: {cls_loss.item():.4f}  "
                  f"link_bpr: {link_loss.item():.4f}  "
                  f"val_cls: {val_cls.item():.4f}")

        if val_cls.item() < best_val_loss:
            best_val_loss  = val_cls.item()
            patience_count = 0
            best_state     = copy.deepcopy(model.state_dict())
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.4f}")

    plt.figure(figsize=(8, 5))
    ep = range(1, len(train_losses) + 1)
    plt.plot(ep, train_losses, label='Train Loss')
    plt.plot(ep, val_losses,   label='Val Loss', linestyle='--')
    plt.title(f'{label} - Training & Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    return model


# Standard Evaluation 

def evaluate_gat(model, data, top_k=TOP_K, label="GATv2"):
    model.eval()
    data_dev = data.to(DEVICE)

    with torch.no_grad():
        emb_dict = model(data_dev.x_dict, data_dev.edge_index_dict)

    # Attribute Prediction 
    print(f"\n{'='*65}\n  STAGE 1 - {label}: Attribute Prediction\n{'='*65}")

    outfit_emb_test = emb_dict['outfit'][data_dev['outfit'].test_mask]
    y_test = data_dev['outfit'].y[data_dev['outfit'].test_mask].cpu().numpy()

    with torch.no_grad():
        logits_list = model.predict_attributes(outfit_emb_test)

    Y_pred = np.stack([l.argmax(dim=1).cpu().numpy() for l in logits_list], axis=1)

    print(f"\n  {'Target':<22} {'Accuracy':>10} {'F1-W':>10} {'Precision':>10} {'Recall':>10}")
    print("  " + "-"*65)
    all_f1 = []
    for i, col in enumerate(TARGET_COLS):
        acc  = accuracy_score(y_test[:, i], Y_pred[:, i])
        f1   = f1_score(y_test[:, i], Y_pred[:, i], average='weighted', zero_division=0)
        prec = precision_score(y_test[:, i], Y_pred[:, i], average='weighted', zero_division=0)
        rec  = recall_score(y_test[:, i], Y_pred[:, i], average='weighted', zero_division=0)
        all_f1.append(f1)
        print(f"  {col:<22} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f}")

    exact  = np.mean(np.all(y_test == Y_pred, axis=1))
    hloss  = hamming_loss_multiclass(y_test, Y_pred)
    avg_f1 = np.mean(all_f1)
    print(f"\n  Exact Match Ratio : {exact:.4f}")
    print(f"  Hamming Loss      : {hloss:.4f}")
    print(f"  Avg F1 (all)      : {avg_f1:.4f}")

    # Recommendation Evaluation 
    print(f"\n{'='*65}\n  STAGE 2 - {label}: Recommendation @ K={top_k}\n{'='*65}")

    outfit_embs_np = emb_dict['outfit'].cpu().numpy()
    item_embs_np   = emb_dict['item'].cpu().numpy()

    # pre-normalise items through link_proj_item for cosine scoring
    with torch.no_grad():
        item_embs_t = torch.tensor(item_embs_np, dtype=torch.float32).to(DEVICE)
        item_link   = F.normalize(model.link_proj_item(item_embs_t), dim=-1).cpu().numpy()

    test_idxs        = torch.where(data['outfit'].test_mask)[0].numpy()
    filtered_metrics = defaultdict(list)
    raw_metrics      = defaultdict(list)

    for oidx in test_idxs:
        row      = demo.iloc[oidx]
        gender   = row['gender']
        occasion = row['occasion']
        img_cat  = DEMO_TO_IMAGE_CAT.get(row['clothing_type'], row['clothing_type'])

        # Ground-truth relevant items: catalogue items of matching gender + category
        relevant_ids = set(item_metadata[
            (item_metadata['gender']     == gender) &
            (item_metadata['class_name'] == img_cat)
        ].index.tolist())
        if not relevant_ids:
            continue

        with torch.no_grad():
            o_t    = torch.tensor(outfit_embs_np[oidx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            o_link = F.normalize(model.link_proj_outfit(o_t), dim=-1).cpu().numpy().squeeze()
        scores = item_link @ o_link

        # Filtered ranking
        allowed_cats = list(set([img_cat] + OCCASION_EXTRA.get(occasion, []) + ACCESSORY_CATS))
        cand_mask    = (item_metadata['gender'] == gender) & \
                       (item_metadata['class_name'].isin(allowed_cats))
        cand_idxs    = item_metadata[cand_mask].index.tolist()
        if len(cand_idxs) < top_k:
            cand_idxs = item_metadata[item_metadata['gender'] == gender].index.tolist()

        cand_scores  = scores[cand_idxs]
        top_local    = np.argsort(cand_scores)[::-1][:top_k]
        rec_filtered = [cand_idxs[j] for j in top_local]

        filtered_metrics['precision'].append(precision_at_k(rec_filtered, relevant_ids, top_k))
        filtered_metrics['recall'].append(recall_at_k(rec_filtered, relevant_ids, top_k))
        filtered_metrics['f1_score'].append(f1_at_k(rec_filtered, relevant_ids, top_k))
        filtered_metrics['ndcg'].append(ndcg_at_k(rec_filtered, relevant_ids, top_k))
        filtered_metrics['map'].append(map_at_k(rec_filtered, relevant_ids, top_k))

        # Raw ranking: full catalogue without any category filter
        top_raw = np.argsort(scores)[::-1][:top_k]
        rec_raw = list(top_raw)
        raw_metrics['precision'].append(precision_at_k(rec_raw, relevant_ids, top_k))
        raw_metrics['recall'].append(recall_at_k(rec_raw, relevant_ids, top_k))
        raw_metrics['ndcg'].append(ndcg_at_k(rec_raw, relevant_ids, top_k))
        raw_metrics['map'].append(map_at_k(rec_raw, relevant_ids, top_k))

    print(f"\n  {'Metric':<25} {'Filtered':>12} {'Raw (no filter)':>16}")
    print("  " + "-"*55)
    summary = {}
    for metric, lbl in [('precision', f'Precision@{top_k}'),
                         ('recall',    f'Recall@{top_k}'),
                         ('f1_score',  f'F1-Score@{top_k}'),
                         ('ndcg',      f'NDCG@{top_k}'),
                         ('map',       f'MAP@{top_k}')]:
        f_score = np.mean(filtered_metrics[metric]) if filtered_metrics[metric] else 0.0
        r_score = np.mean(raw_metrics.get(metric, [])) if raw_metrics.get(metric) else 0.0
        summary[metric] = f_score
        f_str = f"{f_score:>12.4f}"
        r_str = f"{r_score:>16.4f}" if metric != 'f1_score' else f"{'--':>16}"
        print(f"  {lbl:<25} {f_str} {r_str}")

    return {
        'all_f1': all_f1, 'avg_f1': avg_f1,
        'exact_match': exact, 'hamming_loss': hloss,
        **summary
    }


# Cold-Start Evaluation 

def evaluate_cold_start(model: nn.Module,
                        data: HeteroData,
                        clip_normed_all: np.ndarray,
                        top_k: int = TOP_K,
                        label: str = "GATv2",
                        knn_k: int = KNN_K) -> dict:

    model.eval()

    pos_src_all = data['outfit', 'wears', 'item'].edge_index[0]
    pos_dst_all = data['outfit', 'wears', 'item'].edge_index[1]

    train_outfit_mask = data['outfit'].train_mask
    test_outfit_mask  = data['outfit'].test_mask

    # Cold items = items seen in test edges but never in training edges
    train_item_idxs = set(pos_dst_all[train_outfit_mask[pos_src_all]].numpy().tolist())
    test_item_idxs  = set(pos_dst_all[test_outfit_mask[pos_src_all]].numpy().tolist())
    cold_item_idxs  = list(test_item_idxs - train_item_idxs)

    print(f"  Total catalogue items                  : {len(item_metadata)}")
    print(f"  Items seen during training             : {len(train_item_idxs)}")
    print(f"  Items referenced in test edges         : {len(test_item_idxs)}")
    print(f"  Cold-start items (0 train interactions): {len(cold_item_idxs)}")

    if len(cold_item_idxs) == 0:
        print("\n No genuinely cold items found. Using random 10% pseudo-cold sample.")
        rng_cs = np.random.default_rng(99)
        cold_item_idxs = rng_cs.choice(
            len(item_metadata),
            size=max(1, len(item_metadata) // 10),
            replace=False
        ).tolist()
        print(f"  Pseudo-cold items selected: {len(cold_item_idxs)}")

    cold_item_set = set(cold_item_idxs)

    #  Warm embeddings full graph, unaugmented
    data_dev = data.to(DEVICE)
    with torch.no_grad():
        warm_emb_dict = model(data_dev.x_dict, data_dev.edge_index_dict)

    outfit_embs_np = warm_emb_dict['outfit'].cpu().numpy()

    # Pre-project all outfit embeddings through link_proj_outfit
    with torch.no_grad():
        o_all_t    = torch.tensor(outfit_embs_np, dtype=torch.float32).to(DEVICE)
        o_all_link = F.normalize(model.link_proj_outfit(o_all_t), dim=-1).cpu().numpy()
        # shape: (N_outfits, EMBED_DIM)

    #  Build ONE augmented graph for all cold items 
    print(f"  Building single augmented graph for all {len(cold_item_idxs)} cold items...")
    aug_data = copy.deepcopy(data)

    all_new_src, all_new_dst = [], []
    for cold_idx in cold_item_idxs:
        cold_feat      = clip_normed_all[cold_idx]   # (512,) item CLIP feat
        neighbour_idxs = dynamic_link_item(cold_feat, clip_normed_all, k=knn_k)
        for n in neighbour_idxs:
            all_new_src.append(n);        all_new_dst.append(cold_idx)
            all_new_src.append(cold_idx); all_new_dst.append(n)

    new_src_t = torch.tensor(all_new_src, dtype=torch.long)
    new_dst_t = torch.tensor(all_new_dst, dtype=torch.long)

    old_ei = aug_data['item', 'cooccurs', 'item'].edge_index
    aug_data['item', 'cooccurs', 'item'].edge_index = torch.cat(
        [old_ei, torch.stack([new_src_t, new_dst_t])], dim=1
    )

    # ONE forward pass — all cold items updated together
    aug_data_dev = aug_data.to(DEVICE)
    with torch.no_grad():
        aug_emb_dict = model(aug_data_dev.x_dict, aug_data_dev.edge_index_dict)

    # Pre-project augmented item embeddings through link_proj_item
    with torch.no_grad():
        aug_items_t    = aug_emb_dict['item'].to(DEVICE)
        aug_items_link = F.normalize(
            model.link_proj_item(aug_items_t), dim=-1
        ).cpu().numpy()    # (N_items, EMBED_DIM)

    # Warm item link embeddings — fallback for CLIP baseline edge cases
    with torch.no_grad():
        warm_items_t    = warm_emb_dict['item'].to(DEVICE)
        warm_items_link = F.normalize(
            model.link_proj_item(warm_items_t), dim=-1
        ).cpu().numpy()    # (N_items, EMBED_DIM)

    # Build cold item lookup by (gender, class_name)
    cold_items_df = item_metadata[item_metadata.index.isin(cold_item_set)].copy()

    # Pre-build outfit -> connected item indices mapping (for CLIP proxy)
    pos_src_np = pos_src_all.numpy()
    pos_dst_np = pos_dst_all.numpy()

    # Only evaluate test outfits that reference at least one cold item via edges
    test_outfit_idxs = torch.where(data['outfit'].test_mask)[0].numpy()
    test_outfit_set  = set(test_outfit_idxs.tolist())
    outfit_to_cold_items = defaultdict(set)
    for o_idx, i_idx in zip(pos_src_np, pos_dst_np):
        if i_idx in cold_item_set and o_idx in test_outfit_set:
            outfit_to_cold_items[int(o_idx)].add(int(i_idx))

    # Metric accumulation
    gnn_metrics  = defaultdict(list)
    clip_metrics = defaultdict(list)
    evaluated    = 0

    for o_idx in test_outfit_idxs:
        # Only process outfits that actually reference a cold item via graph edges
        if int(o_idx) not in outfit_to_cold_items:
            continue

        row     = demo.iloc[o_idx]
        gender  = row['gender']
        img_cat = DEMO_TO_IMAGE_CAT.get(row['clothing_type'], row['clothing_type'])

        # Relevant = cold items of matching gender + clothing category
        relevant = set(
            cold_items_df[
                (cold_items_df['gender']     == gender) &
                (cold_items_df['class_name'] == img_cat)
            ].index.tolist()
        )

        # Fall back to gender match only if no category match in cold pool
        if not relevant:
            relevant = set(
                cold_items_df[cold_items_df['gender'] == gender].index.tolist()
            )

        # Skip if still empty
        if not relevant:
            continue

        # GNN scores: augmented embeddings vs warm outfit projection 
        scores_gnn = aug_items_link @ o_all_link[o_idx]   # (N_items,)
        top_gnn    = list(np.argsort(scores_gnn)[::-1][:top_k])
        gnn_metrics['precision'].append(precision_at_k(top_gnn, relevant, top_k))
        gnn_metrics['recall'].append(recall_at_k(top_gnn, relevant, top_k))
        gnn_metrics['ndcg'].append(ndcg_at_k(top_gnn, relevant, top_k))
        gnn_metrics['map'].append(map_at_k(top_gnn, relevant, top_k))

        # CLIP baseline: mean CLIP of connected warm items as outfit proxy 
        connected_mask      = (pos_src_np == o_idx)
        connected_item_idxs = pos_dst_np[connected_mask]

        if len(connected_item_idxs) > 0:
            outfit_clip_proxy = clip_normed_all[connected_item_idxs].mean(axis=0)  # (512,)
            norm              = np.linalg.norm(outfit_clip_proxy) + 1e-8
            outfit_clip_proxy = outfit_clip_proxy / norm
            scores_clip       = clip_normed_all @ outfit_clip_proxy   # (N_items,)
        else:
            # Fallback: no connected warm items — use learned link space
            scores_clip = warm_items_link @ o_all_link[o_idx]

        top_clip = list(np.argsort(scores_clip)[::-1][:top_k])
        clip_metrics['precision'].append(precision_at_k(top_clip, relevant, top_k))
        clip_metrics['recall'].append(recall_at_k(top_clip, relevant, top_k))
        clip_metrics['ndcg'].append(ndcg_at_k(top_clip, relevant, top_k))
        clip_metrics['map'].append(map_at_k(top_clip, relevant, top_k))

        evaluated += 1

    if evaluated == 0:
        print("\n  WARNING: No (outfit, cold-item) test pairs found.")
        return {}

    print(f"\n  Evaluated {evaluated} (outfit, cold-item) test pairs")
    print(f"\n  {'Metric':<25} {'KNN-GNN (Ours)':>16} {'CLIP Baseline':>16} {'Delta':>10}")
    print("  " + "-"*72)

    cs_summary = {}
    for metric, lbl in [('precision', f'Precision@{top_k}'),
                         ('recall',    f'Recall@{top_k}'),
                         ('ndcg',      f'nDCG@{top_k}'),
                         ('map',       f'MAP@{top_k}')]:
        gnn_score  = np.mean(gnn_metrics[metric])
        clip_score = np.mean(clip_metrics[metric])
        delta      = (gnn_score - clip_score) / (clip_score + 1e-8) * 100
        cs_summary[f'cs_{metric}_gnn']  = gnn_score
        cs_summary[f'cs_{metric}_clip'] = clip_score
        print(f"  {lbl:<25} {gnn_score:>16.4f} {clip_score:>16.4f} {delta:>+9.1f}%")

    return cs_summary


# Main 
# Train WITH AGE 
model_with_age = train_gat(
    data, outfit_dim=outfit_features.shape[1],
    label="GATv2 (WITH AGE)", epochs=EPOCHS
)
torch.save(model_with_age.state_dict(), SAVE_PATH_WITH_AGE)
print(f"  Model saved: {SAVE_PATH_WITH_AGE}")

results_with_age = evaluate_gat(
    model_with_age, data, top_k=TOP_K, label="GATv2 (WITH AGE)"
)

# Cold-Start Evaluation 
cold_start_results = evaluate_cold_start(
    model           = model_with_age,
    data            = data,
    clip_normed_all = clip_normed,
    top_k           = TOP_K,
    label           = "GATv2 (WITH AGE)",
    knn_k           = KNN_K
)

#  Train without AGE 
print("\n" + "="*65 + "\n  ABLATION: BUILDING GRAPH WITHOUT AGE\n" + "="*65)

data_na = HeteroData()
data_na['outfit'].x          = torch.tensor(outfit_features_no_age, dtype=torch.float32)
data_na['outfit'].y          = data['outfit'].y.clone()
data_na['outfit'].train_mask = data['outfit'].train_mask.clone()
data_na['outfit'].val_mask   = data['outfit'].val_mask.clone()
data_na['outfit'].test_mask  = data['outfit'].test_mask.clone()
data_na['item'].x            = data['item'].x.clone()

for etype in data.edge_types:
    if 'same_age' in str(etype):
        continue   # remove age-homophily edges for ablation
    data_na[etype].edge_index = data[etype].edge_index.clone()

data_na = T.ToUndirected()(data_na)
print(f"  Outfit features (no age): {data_na['outfit'].x.shape}  "
      f"(vs {data['outfit'].x.shape} with age - "
      f"{outfit_features.shape[1] - outfit_features_no_age.shape[1]} age dims removed)")

model_no_age = train_gat(
    data_na, outfit_dim=outfit_features_no_age.shape[1],
    label="GATv2 (WITHOUT AGE)", epochs=EPOCHS
)
torch.save(model_no_age.state_dict(), SAVE_PATH_NO_AGE)

results_no_age = evaluate_gat(
    model_no_age, data_na, top_k=TOP_K, label="GATv2 (WITHOUT AGE)"
)

# Ablation Summary with and without age
print("  FINAL ABLATION SUMMARY - WITH AGE vs WITHOUT AGE (GATv2)")

print(f"\n  Stage 1 : Attribute Prediction")
print(f"  {'Metric':<28} {'GAT+Age':>12} {'GAT-Age':>12} {'Delta Drop':>12}")
print("  " + "-"*67)
for i, col in enumerate(TARGET_COLS):
    wa   = results_with_age['all_f1'][i]
    na   = results_no_age['all_f1'][i]
    drop = (wa - na) / (wa + 1e-8) * 100
    print(f"  F1 - {col:<22} {wa:>12.4f} {na:>12.4f} {drop:>+11.2f}%")
print("  " + "-"*67)

wa_avg = results_with_age['avg_f1'];       na_avg = results_no_age['avg_f1']
wa_em  = results_with_age['exact_match'];  na_em  = results_no_age['exact_match']
wa_hl  = results_with_age['hamming_loss']; na_hl  = results_no_age['hamming_loss']
print(f"  {'Avg F1':<28} {wa_avg:>12.4f} {na_avg:>12.4f} "
      f"{(wa_avg - na_avg) / (wa_avg + 1e-8) * 100:>+11.2f}%")
print(f"  {'Exact Match':<28} {wa_em:>12.4f} {na_em:>12.4f} "
      f"{(wa_em - na_em) / (wa_em + 1e-8) * 100:>+11.2f}%")
print(f"  {'Hamming Loss':<28} {wa_hl:>12.4f} {na_hl:>12.4f} "
      f"  (error increases w/o age)")

print(f"\n  Stage 2 : Recommendation Quality (@ K={TOP_K})")
print(f"  {'Metric':<28} {'GAT+Age':>12} {'GAT-Age':>12} {'Delta Drop':>12}")
print("  " + "-"*67)
for metric, lbl in [('precision', f'Precision@{TOP_K}'),
                    ('recall',    f'Recall@{TOP_K}'),
                    ('f1_score',  f'F1-Score@{TOP_K}'),
                    ('ndcg',      f'NDCG@{TOP_K}'),
                    ('map',       f'MAP@{TOP_K}')]:
    wa   = results_with_age[metric]
    na   = results_no_age[metric]
    drop = (wa - na) / (wa + 1e-8) * 100
    print(f"  {lbl:<28} {wa:>12.4f} {na:>12.4f} {drop:>+11.2f}%")

#  Cold-Start Summary 
if cold_start_results:
    print(f"\n{'='*70}")
    print(f"  COLD-START SUMMARY - KNN-GNN vs CLIP Baseline")
    print(f"  Positive Delta = graph structure adds value beyond raw visual similarity")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'KNN-GNN':>12} {'CLIP Base':>12} {'Delta Gain':>12}")
    print("  " + "-"*59)
    for metric in ['precision', 'recall', 'ndcg', 'map']:
        gnn_val  = cold_start_results.get(f'cs_{metric}_gnn',  0.0)
        clip_val = cold_start_results.get(f'cs_{metric}_clip', 0.0)
        delta    = (gnn_val - clip_val) / (clip_val + 1e-8) * 100
        lbl      = f'{metric.upper()}@{TOP_K}'
        print(f"  {lbl:<20} {gnn_val:>12.4f} {clip_val:>12.4f} {delta:>+11.1f}%")
