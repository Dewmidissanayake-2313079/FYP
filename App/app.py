import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from PIL import Image
import os
import yaml
import warnings
from dotenv import load_dotenv
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ultralytics import YOLO
import clip
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv
import torch_geometric.transforms as T
import google.generativeai as genai
from google import genai as genai_new
from google.genai import types as genai_types

# Load secret environment variables
load_dotenv()

# Load application configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    _cfg = yaml.safe_load(f)

warnings.filterwarnings('ignore')

class Config:
    # Paths
    YOLO_WEIGHTS    = _cfg['paths']['yolo_weights']
    GAT_WEIGHTS     = _cfg['paths']['gat_weights']
    DATASET_PATH    = _cfg['paths']['dataset_path']
    METADATA_PATH   = _cfg['paths']['metadata_path']
    CLIP_FEAT_PATH  = _cfg['paths']['clip_feat_path']
    IMAGE_DIR       = _cfg['paths']['image_dir']
    CROPS_DIR       = _cfg['paths']['crops_dir']
    OUTFIT_EMB_PATH = _cfg['paths']['outfit_emb_path']
    ITEM_EMB_PATH   = _cfg['paths']['item_emb_path']
    WEARS_EDGES_PATH= _cfg['paths']['wears_edges_path']

    # API Keys (Loaded in .env)
    GEMINI_API_KEY   = os.getenv('GEMINI_API_KEY')
    NANOBANANA_TOKEN = os.getenv('NANOBANANA_TOKEN')

    # Model Hyperparameters
    HIDDEN_DIM   = _cfg['model']['hidden_dim']
    EMBED_DIM    = _cfg['model']['embed_dim']
    NUM_LAYERS   = _cfg['model']['num_layers']
    NUM_HEADS    = _cfg['model']['num_heads']
    DROPOUT      = _cfg['model']['dropout']
    ATTN_DROPOUT = _cfg['model']['attn_dropout']

    # Operational Parameters
    TOP_K            = _cfg['parameters']['top_k']
    SIMILAR_COUNT    = _cfg['parameters']['similar_count']
    COMPLEMENT_COUNT = _cfg['parameters']['complement_count']
    ACCESSORY_COUNT  = _cfg['parameters']['accessory_count']
    YOLO_CONF        = _cfg['parameters']['yolo_conf']
    CLIP_RERANK_POOL = _cfg['parameters']['clip_rerank_pool']

    # Recommendation Engine Weights
    SIMILAR_WEIGHTS    = _cfg['weights']['similar']
    COMPLEMENT_WEIGHTS = _cfg['weights']['complementary']
    ACCESSORY_WEIGHTS  = _cfg['weights']['accessory']

    # System Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Domain Mappings
    AGE_GROUP_MAP          = _cfg['mappings']['age_group_map']
    DEMO_TO_IMAGE_CAT      = _cfg['mappings']['demo_to_image_cat']
    OCCASION_CATEGORIES    = _cfg['mappings']['occasion_categories']
    ACCESSORY_CATS         = _cfg['mappings']['accessory_cats']
    IMAGE_NAME_TO_CATEGORY = _cfg['mappings']['image_name_to_category']
    COMPLEMENT_MAP         = _cfg['mappings']['complement_map']



# GAT model initialization
# Homogeneous edge types — eligible for self-loops
HOMO_EDGE_TYPES = {
    ('outfit', 'similar',  'outfit'),
    ('outfit', 'same_age', 'outfit'),
    ('item',   'cooccurs', 'item'),
}

class HeteroGATv2(nn.Module):

    def __init__(self, outfit_dim, item_dim, hidden_dim, embed_dim,
                 num_classes_list, num_layers=2, num_heads=4,
                 dropout=0.3, attn_dropout=0.1, dropedge=0.1):
        super().__init__()
        self.dropout    = dropout
        self.dropedge   = dropedge
        self.num_layers = num_layers
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads

        self.outfit_proj = nn.Sequential(
            nn.Linear(outfit_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.item_proj = nn.Sequential(
            nn.Linear(item_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())

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
        # Cosine similarity
        o = F.normalize(self.link_proj_outfit(outfit_emb), dim=-1)
        i = F.normalize(self.link_proj_item(item_emb),   dim=-1)
        return (o * i).sum(dim=-1)



# Building the graph (GAT)
class GraphBuilder:

    def __init__(self, config):
        self.config = config
        self.demo = pd.read_csv(config.DATASET_PATH)
        self.item_metadata = pd.read_csv(config.METADATA_PATH)
        self.clip_features = np.load(config.CLIP_FEAT_PATH).astype(np.float32)

        if 'clothing_type' not in self.demo.columns:
            self.demo['clothing_type'] = self.demo.apply(self._derive_clothing_type, axis=1)
        self.demo['colour_bottom'] = self.demo['colour_bottom'].fillna('unknown')

        self.item_metadata['gender'] = self.item_metadata['image_name'].apply(
            lambda x: 'male' if 'MEN' in x.upper() and 'WOMEN' not in x.upper() else 'female')

        self.item_metadata['true_category'] = self.item_metadata['image_name'].apply(
            self._parse_true_category)

        self._encode_features()

    @staticmethod
    def _derive_clothing_type(row):
        if pd.notna(row.get('full_body', np.nan)):
            if row['full_body'] == 'dress': return 'dress'
            if row['full_body'] in ['play_suit', 'jump_suit']: return 'rompers'
        lb = row.get('lower_body', np.nan)
        if pd.notna(lb):
            if lb == 'leggings': return 'leggings'
            if lb == 'skirt': return 'skirt'
            if lb in ['pants/trousers', 'jeans', 'shorts', 'athletic_pants']: return 'pants'
        return 'top'

    @staticmethod
    def _parse_true_category(image_name):
        parts = image_name.split('-')
        if len(parts) >= 2:
            raw_cat = parts[1]
            if raw_cat in Config.IMAGE_NAME_TO_CATEGORY:
                return Config.IMAGE_NAME_TO_CATEGORY[raw_cat]
            for key, val in Config.IMAGE_NAME_TO_CATEGORY.items():
                if key.lower() in raw_cat.lower():
                    return val
        return 'unknown'

    def _encode_features(self):
        self.target_cols = ['clothing_type', 'colour_top', 'colour_bottom',
                            'neckline', 'sleeve_length']
        self.target_encoders = {}
        self.num_classes = {}
        for col in self.target_cols:
            le = LabelEncoder()
            self.demo[f'{col}_enc'] = le.fit_transform(self.demo[col])
            self.target_encoders[col] = le
            self.num_classes[col] = len(le.classes_)

        self.age_group_le = LabelEncoder().fit(self.demo['age_group'])
        self.gender_le    = LabelEncoder().fit(self.demo['gender'])
        self.occasion_le  = LabelEncoder().fit(self.demo['occasion'])
        self.demo['age_group_enc'] = self.age_group_le.transform(self.demo['age_group'])
        self.demo['gender_enc']    = self.gender_le.transform(self.demo['gender'])
        self.demo['occasion_enc']  = self.occasion_le.transform(self.demo['occasion'])

        n_ag  = len(self.age_group_le.classes_)
        n_g   = len(self.gender_le.classes_)
        n_occ = len(self.occasion_le.classes_)
        self.outfit_dim = n_ag + n_g + n_occ

        norms = np.linalg.norm(self.clip_features, axis=1, keepdims=True) + 1e-8
        self.item_clip_normed = (self.clip_features / norms).astype(np.float32)

        self.class_le = LabelEncoder()
        self.item_metadata['class_enc'] = self.class_le.fit_transform(
            self.item_metadata['class_name'])

# Clip utility initializations
@torch.no_grad()
def extract_clip_features_batch(pil_images, clip_model, clip_preprocess, device):
    tensors = torch.stack([clip_preprocess(img) for img in pil_images]).to(device)
    features = clip_model.encode_image(tensors)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


# Loading the cached models
@st.cache_resource(show_spinner="Loading models and data ...")
def load_system():
    cfg = Config

    yolo_model = YOLO(cfg.YOLO_WEIGHTS)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=cfg.DEVICE)
    clip_model.eval()

    graph_builder = GraphBuilder(cfg)

    nc_list = [graph_builder.num_classes[c] for c in graph_builder.target_cols]
    model = HeteroGATv2(
        outfit_dim=graph_builder.outfit_dim, item_dim=512,
        hidden_dim=cfg.HIDDEN_DIM, embed_dim=cfg.EMBED_DIM,
        num_classes_list=nc_list, num_layers=cfg.NUM_LAYERS,
        num_heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT,
        attn_dropout=cfg.ATTN_DROPOUT,
    ).to(cfg.DEVICE)

    state_dict = torch.load(cfg.GAT_WEIGHTS, map_location=cfg.DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    item_embeddings   = np.load(cfg.ITEM_EMB_PATH).astype(np.float32)
    outfit_embeddings = np.load(cfg.OUTFIT_EMB_PATH).astype(np.float32)
    wears_edges       = np.load(cfg.WEARS_EDGES_PATH)

    item_emb_norm = item_embeddings / (np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-8)
    outfit_emb_norm = outfit_embeddings / (np.linalg.norm(outfit_embeddings, axis=1, keepdims=True) + 1e-8)

    return {
        'yolo': yolo_model,
        'clip_model': clip_model,
        'clip_preprocess': clip_preprocess,
        'model': model,
        'graph_builder': graph_builder,
        'wears_edges': wears_edges,
        'item_embeddings': item_embeddings,
        'item_emb_norm': item_emb_norm,
        'outfit_emb_norm': outfit_emb_norm,
    }

# Recommendation engine
def detect_items(yolo_model, pil_image, conf=0.25):
    img_array = np.array(pil_image)
    results = yolo_model(img_array, conf=conf, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for i in range(len(result.boxes)):
            bbox = result.boxes.xyxy[i].cpu().numpy().astype(int)
            cls_id = int(result.boxes.cls[i].cpu())
            conf_val = float(result.boxes.conf[i].cpu())
            cls_name = yolo_model.names[cls_id]
            x1, y1, x2, y2 = bbox
            crop = pil_image.crop((x1, y1, x2, y2))
            area = (x2 - x1) * (y2 - y1)
            detections.append({
                'bbox': bbox.tolist(),
                'class_name': cls_name,
                'confidence': conf_val,
                'crop': crop,
                'area': area,
            })
    return detections


def build_query_embedding(sys, age_group, gender, occasion, user_clip_features=None):
    gb = sys['graph_builder']
    model = sys['model']
    cfg = Config

    resolved_age = cfg.AGE_GROUP_MAP.get(age_group, age_group)

    n_ag  = len(gb.age_group_le.classes_)
    n_g   = len(gb.gender_le.classes_)
    n_occ = len(gb.occasion_le.classes_)

    feat = np.zeros(n_ag + n_g + n_occ, dtype=np.float32)
    if resolved_age in gb.age_group_le.classes_:
        feat[gb.age_group_le.transform([resolved_age])[0]] = 1.0
    if gender in gb.gender_le.classes_:
        feat[n_ag + gb.gender_le.transform([gender])[0]] = 1.0
    if occasion in gb.occasion_le.classes_:
        feat[n_ag + n_g + gb.occasion_le.transform([occasion])[0]] = 1.0

    with torch.no_grad():
        feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)
        hidden = model.outfit_proj(feat_tensor)
        emb_raw = model.embed_proj(hidden)
        # Apply link projection + L2 normalization
        demo_emb = F.normalize(model.link_proj_outfit(emb_raw), dim=-1).cpu().numpy().flatten()

    if user_clip_features is not None and len(user_clip_features) > 0:
        if user_clip_features.ndim == 1:
            user_clip_features = user_clip_features.reshape(1, -1)
        user_clip_avg = user_clip_features.mean(axis=0)
        user_clip_avg = user_clip_avg / (np.linalg.norm(user_clip_avg) + 1e-8)
        sims = gb.item_clip_normed @ user_clip_avg
        top_visual_idxs = np.argsort(sims)[::-1][:20]
        # Item embeddings are already normalized via link_proj_item in the .npy files
        visual_gnn_emb = sys['item_embeddings'][top_visual_idxs].mean(axis=0)
        visual_gnn_emb = visual_gnn_emb / (np.linalg.norm(visual_gnn_emb) + 1e-8)
    else:
        visual_gnn_emb = np.zeros_like(demo_emb)

    alpha = 0.15   # reduced: demo_emb (age/gender/occasion) now dominates the query
    fused = alpha * visual_gnn_emb + (1 - alpha) * demo_emb
    fused = fused / (np.linalg.norm(fused) + 1e-8)
    return fused


def get_demographic_item_scores(sys, age_group, gender, occasion):
    gb = sys['graph_builder']
    demo = gb.demo
    wears_edges = sys['wears_edges']
    n_items = len(gb.item_metadata)

    resolved_age = Config.AGE_GROUP_MAP.get(age_group, age_group)

    mask = (demo['gender'] == gender)
    if occasion in demo['occasion'].values:
        mask_occ = mask & (demo['occasion'] == occasion)
        if mask_occ.sum() > 0: mask = mask_occ
    if resolved_age in demo['age_group'].values:
        mask_age = mask & (demo['age_group'] == resolved_age)
        if mask_age.sum() > 0: mask = mask_age

    matching_outfits = set(demo[mask].index.tolist())
    if not matching_outfits:
        return np.zeros(n_items, dtype=np.float32)

    edge_src = wears_edges[0]
    edge_dst = wears_edges[1]

    item_counts = np.zeros(n_items, dtype=np.float32)
    for src, dst in zip(edge_src, edge_dst):
        if src in matching_outfits:
            item_counts[dst] += 1.0

    max_count = item_counts.max()
    if max_count > 0:
        item_counts /= max_count
    return item_counts


def score_candidates(sys, query_emb, user_clip_avg, candidate_idxs,
                     demo_scores=None, class_name_boost=None,
                     weights=None):

    cfg = Config
    gb = sys['graph_builder']

    # Use provided weights or fall back to balanced defaults
    if weights is None:
        weights = {'clip': 0.35, 'gnn': 0.40, 'demo': 0.15, 'label': 0.10}

    cand_gnn  = sys['item_emb_norm'][candidate_idxs]
    cand_clip = gb.item_clip_normed[candidate_idxs]

    gnn_scores = cand_gnn @ query_emb

    if user_clip_avg is not None:
        clip_scores = cand_clip @ user_clip_avg
    else:
        clip_scores = np.zeros(len(candidate_idxs), dtype=np.float32)

    if demo_scores is not None:
        demo_s = demo_scores[candidate_idxs]
    else:
        demo_s = np.zeros(len(candidate_idxs), dtype=np.float32)

    if class_name_boost:
        md = gb.item_metadata
        label_match = np.array([
            1.0 if md.iloc[idx]['true_category'] == class_name_boost else 0.0
            for idx in candidate_idxs
        ], dtype=np.float32)
    else:
        label_match = np.zeros(len(candidate_idxs), dtype=np.float32)

    final_scores = (weights['gnn'] * gnn_scores +
                    weights['clip'] * clip_scores +
                    weights['demo'] * demo_s +
                    weights['label'] * label_match)
    return final_scores


def find_similar_with_clip_rerank(sys, query_emb, primary_clip_vec, 
                                   similar_pool, demo_scores, primary_img_cat,
                                   count):
    cfg = Config
    gb = sys['graph_builder']
    md = gb.item_metadata

    if not similar_pool or primary_clip_vec is None:
        return []

    # Age-aware pre-filtering: blend visual + demographic relevance
    pool_clip = gb.item_clip_normed[similar_pool]
    clip_scores = pool_clip @ primary_clip_vec

    # Normalise CLIP scores to [0, 1]
    c_min, c_max = clip_scores.min(), clip_scores.max()
    clip_norm = (clip_scores - c_min) / (c_max - c_min + 1e-8)

    # Normalise demographic scores for this pool
    if demo_scores is not None:
        pool_demo = demo_scores[np.array(similar_pool)]
        d_min, d_max = pool_demo.min(), pool_demo.max()
        demo_norm = (pool_demo - d_min) / (d_max - d_min + 1e-8)
    else:
        demo_norm = np.zeros(len(similar_pool), dtype=np.float32)

    # 65% visual + 35% age/demographic — ensures age-relevant items enter the shortlist
    blend_scores = 0.65 * clip_norm + 0.35 * demo_norm

    n_prefilter = min(cfg.CLIP_RERANK_POOL, len(similar_pool))
    top_blend_local = np.argsort(blend_scores)[::-1][:n_prefilter]
    shortlist_idxs = [similar_pool[j] for j in top_blend_local]

    #  Re-rank shortlist with full hybrid scoring 
    hybrid_scores = score_candidates(
        sys, query_emb, primary_clip_vec, shortlist_idxs,
        demo_scores=demo_scores,
        class_name_boost=primary_img_cat,
        weights=cfg.SIMILAR_WEIGHTS,
    )

    # Deduplicate by image base name and select top 
    sorted_hybrid = np.argsort(hybrid_scores)[::-1]
    results = []
    seen_images = set()

    for idx_pos in sorted_hybrid:
        item_idx = shortlist_idxs[idx_pos]
        img_name = md.iloc[item_idx]['image_name']
        # Deduplicate: same garment from different angles
        base_name = img_name.rsplit('-', 1)[0] if '-' in img_name else img_name
        if base_name not in seen_images:
            results.append({
                'idx': item_idx,
                'score': hybrid_scores[idx_pos],
                'clip_score': float(gb.item_clip_normed[item_idx] @ primary_clip_vec),
                'class': md.iloc[item_idx]['class_name'],
                'true_cat': md.iloc[item_idx]['true_category'],
                'image_name': img_name,
            })
            seen_images.add(base_name)
        if len(results) >= count:
            break

    return results


def get_item_image(item_idx, sys):
    md = sys['graph_builder'].item_metadata
    row = md.iloc[item_idx]
    full_path = os.path.join(Config.IMAGE_DIR, f"{row['image_name']}.jpg")
    if os.path.exists(full_path):
        return Image.open(full_path).convert('RGB')
    return None


def run_recommendation(sys, uploaded_images, age_group, gender, occasion):
    cfg = Config
    gb = sys['graph_builder']
    md = gb.item_metadata

    # Detect items from all uploaded images
    all_detections = []
    all_clip_feats = []

    for pil_img in uploaded_images:
        dets = detect_items(sys['yolo'], pil_img, conf=cfg.YOLO_CONF)
        all_detections.extend(dets)
        if dets:
            crops = [d['crop'] for d in dets]
            feats = extract_clip_features_batch(
                crops, sys['clip_model'], sys['clip_preprocess'], cfg.DEVICE)
            all_clip_feats.append(feats)

    if all_clip_feats:
        user_clip_feats = np.vstack(all_clip_feats)
        user_clip_avg = user_clip_feats.mean(axis=0)
        user_clip_avg = user_clip_avg / (np.linalg.norm(user_clip_avg) + 1e-8)
    else:
        user_clip_feats = None
        user_clip_avg = None

    # Per-detection CLIP by category
    per_detection_clips = {}
    if all_detections and all_clip_feats:
        all_feats_flat = np.vstack(all_clip_feats)
        cat_feats = defaultdict(list)
        feat_idx = 0
        for det in all_detections:
            if feat_idx < len(all_feats_flat):
                cat_feats[det['class_name']].append(all_feats_flat[feat_idx])
                feat_idx += 1
        for cat, feats in cat_feats.items():
            avg = np.mean(feats, axis=0)
            per_detection_clips[cat] = avg / (np.linalg.norm(avg) + 1e-8)

    # Build query embedding 
    query_emb = build_query_embedding(sys, age_group, gender, occasion, user_clip_feats)

    # Predict attributes 
    with torch.no_grad():
        q_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)
        attr_logits = sys['model'].predict_attributes(q_tensor)
        predicted_attrs = {}
        for i, col in enumerate(gb.target_cols):
            pred_idx = attr_logits[i].argmax(dim=1).item()
            predicted_attrs[col] = gb.target_encoders[col].inverse_transform([pred_idx])[0]

    # Primary category by largest bounding box area 
    detections_sorted = sorted(all_detections, key=lambda d: d['area'], reverse=True)
    main_detections = [d for d in detections_sorted if d['class_name'] not in cfg.ACCESSORY_CATS]

    if main_detections:
        primary_cat = main_detections[0]['class_name']
    else:
        primary_cat = predicted_attrs.get('clothing_type', 'top')

    primary_img_cat = cfg.DEMO_TO_IMAGE_CAT.get(primary_cat, primary_cat)

    #  Build category pools using true_category 
    demo_scores = get_demographic_item_scores(sys, age_group, gender, occasion)

    complement_cats = cfg.COMPLEMENT_MAP.get(primary_img_cat, ['top', 'pants', 'dress'])
    occ_cats = cfg.OCCASION_CATEGORIES.get(occasion, [])
    complement_cats = list(set(complement_cats + [c for c in occ_cats 
                                                   if c != primary_img_cat
                                                   and c not in cfg.ACCESSORY_CATS]))

    accessory_cats_set = set(cfg.ACCESSORY_CATS)

    similar_pool = md[
        (md['gender'] == gender) & (md['true_category'] == primary_img_cat)
    ].index.tolist()

    comp_pool = md[
        (md['gender'] == gender) &
        (md['true_category'].isin(complement_cats)) &
        (~md['true_category'].isin(accessory_cats_set))
    ].index.tolist()

    acc_pool = md[
        (md['gender'] == gender) &
        (md['true_category'].isin(accessory_cats_set))
    ].index.tolist()

    results = {
        'detections': all_detections,
        'predicted_attrs': predicted_attrs,
        'primary_category': primary_img_cat,
        'query_emb': query_emb,
        'user_clip_avg': user_clip_avg,
        'demo_scores': demo_scores,
    }


    primary_clip_vec = per_detection_clips.get(primary_cat, user_clip_avg)
    results['similar'] = find_similar_with_clip_rerank(
        sys, query_emb, primary_clip_vec,
        similar_pool, demo_scores, primary_img_cat,
        count=cfg.SIMILAR_COUNT,
    )

    # Complementary items GNN+demographic weighted 
    if comp_pool:
        similar_selected = {r['idx'] for r in results['similar']}
        comp_scores = score_candidates(
            sys, query_emb, user_clip_avg, comp_pool,
            demo_scores=demo_scores,
            weights=cfg.COMPLEMENT_WEIGHTS,
        )
        true_cats = md.iloc[comp_pool]['true_category'].values
        for idx_pos in range(len(comp_pool)):
            if true_cats[idx_pos] in complement_cats:
                comp_scores[idx_pos] += cfg.COMPLEMENT_WEIGHTS['label']
            if comp_pool[idx_pos] in similar_selected:
                comp_scores[idx_pos] = -999.0

        sorted_comp = np.argsort(comp_scores)[::-1]
        selected_comp = []
        seen_types = set()
        seen_bases = set()
        for j in sorted_comp:
            ctype = true_cats[j]
            img_name = md.iloc[comp_pool[j]]['image_name']
            base_name = img_name.rsplit('-', 1)[0] if '-' in img_name else img_name
            if base_name not in seen_bases and (ctype not in seen_types or len(selected_comp) < cfg.COMPLEMENT_COUNT):
                selected_comp.append(j)
                seen_types.add(ctype)
                seen_bases.add(base_name)
            if len(selected_comp) >= cfg.COMPLEMENT_COUNT:
                break

        results['complementary'] = [
            {'idx': comp_pool[j], 'score': comp_scores[j],
             'class': md.iloc[comp_pool[j]]['class_name'],
             'true_cat': md.iloc[comp_pool[j]]['true_category'],
             'image_name': md.iloc[comp_pool[j]]['image_name']}
            for j in selected_comp
        ]
    else:
        results['complementary'] = []

    # Accessories recommendation
    if acc_pool:
        acc_scores = score_candidates(
            sys, query_emb, user_clip_avg, acc_pool,
            demo_scores=demo_scores,
            weights=cfg.ACCESSORY_WEIGHTS,
        )
        acc_true_cats = md.iloc[acc_pool]['true_category'].values
        sorted_acc = np.argsort(acc_scores)[::-1]
        selected_acc = []
        seen_acc = set()
        seen_acc_bases = set()
        for j in sorted_acc:
            atype = acc_true_cats[j]
            img_name = md.iloc[acc_pool[j]]['image_name']
            base_name = img_name.rsplit('-', 1)[0] if '-' in img_name else img_name
            if base_name not in seen_acc_bases and (atype not in seen_acc or len(selected_acc) < cfg.ACCESSORY_COUNT):
                selected_acc.append(j)
                seen_acc.add(atype)
                seen_acc_bases.add(base_name)
            if len(selected_acc) >= cfg.ACCESSORY_COUNT:
                break

        results['accessories'] = [
            {'idx': acc_pool[j], 'score': acc_scores[j],
             'class': md.iloc[acc_pool[j]]['class_name'],
             'true_cat': md.iloc[acc_pool[j]]['true_category'],
             'image_name': md.iloc[acc_pool[j]]['image_name']}
            for j in selected_acc
        ]
    else:
        results['accessories'] = []

    return results

# XAI and Virtual Try-On 

genai.configure(api_key=Config.GEMINI_API_KEY)

def generate_outfit_explanation(age, gender, occasion, item_details):
    """Generates a friendly AI explanation for why a recommended item fits the user."""
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    You are a friendly personal stylist. Write 2-3 short, warm sentences explaining why this clothing item is a great pick for this person.

    Person: {gender}, age group {age}, heading to a {occasion} event.
    Item recommended: {item_details['true_cat'].replace('_', ' ')}

    Keep it simple, positive, and easy to understand. No technical terms.
    Focus on: why it suits the occasion, how it fits their style, and why they will love it.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "This item is a great match for your style and the occasion — you'll love it!"

#  Visual XAI 

def draw_yolo_detections(pil_image, detections):
    """Overlay YOLO bounding boxes on the uploaded image."""
    COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(pil_image)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = COLORS[i % len(COLORS)]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 6, 0),
                f"{det['class_name']} {det['confidence']:.2f}",
                color='white', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
    ax.axis('off')
    ax.set_title('Detected Fashion Items (YOLO)', fontsize=10, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_score_breakdown(item_idx, query_emb, user_clip_avg, demo_scores, sys_data):
    """Horizontal bar chart showing GNN / CLIP / Demographic score contributions."""
    gb = sys_data['graph_builder']
    gnn_s  = float(sys_data['item_emb_norm'][item_idx] @ query_emb)
    clip_s = (float(gb.item_clip_normed[item_idx] @ user_clip_avg)
              if user_clip_avg is not None else 0.0)
    demo_s = float(demo_scores[item_idx]) if demo_scores is not None else 0.0

    # Cosine similarity [-1,1] → [0,1]
    gnn_s  = max(0.0, min(1.0, (gnn_s  + 1.0) / 2.0))
    clip_s = max(0.0, min(1.0, (clip_s + 1.0) / 2.0))
    demo_s = max(0.0, min(1.0, demo_s))

    labels = ['GNN Graph\nCompatibility', 'Visual CLIP\nSimilarity', 'Demographic\nMatch']
    values = [gnn_s, clip_s, demo_s]
    colors = ['#6C63FF', '#48CAE4', '#F77F00']

    fig, ax = plt.subplots(figsize=(4, 2.4))
    bars = ax.barh(labels, values, color=colors, height=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Score (0 – 1)', fontsize=8)
    ax.set_title('Why was this recommended?', fontsize=9, fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(min(val + 0.03, 0.95), bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


#  Graph XAI 

def get_graph_explanation(item_idx, age_group, gender, occasion, sys_data):

    gb          = sys_data['graph_builder']
    demo        = gb.demo
    wears_edges = sys_data['wears_edges']
    cfg         = Config

    resolved_age = cfg.AGE_GROUP_MAP.get(age_group, age_group)

    # Identify demographically similar outfit nodes
    mask = (demo['gender'] == gender)
    m2 = mask & (demo['occasion'] == occasion)
    if m2.sum() > 0: mask = m2
    if resolved_age in demo['age_group'].values:
        m3 = mask & (demo['age_group'] == resolved_age)
        if m3.sum() > 0: mask = m3

    similar_idxs = set(demo[mask].index.tolist())

    # Count how many similar-profile outfits contain this item
    segment_counts = {}
    for src, dst in zip(wears_edges[0], wears_edges[1]):
        if dst == item_idx and src in similar_idxs and src < len(demo):
            row = demo.iloc[src]
            seg = f"{row['gender'].title()} · {row['age_group']} · {row['occasion']}"
            segment_counts[seg] = segment_counts.get(seg, 0) + 1

    top_segments = dict(
        sorted(segment_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    )

    # Determine which weight set was used for this item
    item_true_cat = gb.item_metadata.iloc[item_idx]['true_category']
    if item_true_cat in cfg.ACCESSORY_CATS:
        w, label = cfg.ACCESSORY_WEIGHTS, 'Accessory'
    elif item_true_cat in cfg.COMPLEMENT_MAP.get(
            cfg.OCCASION_CATEGORIES.get(occasion, ['top'])[0], []):
        w, label = cfg.COMPLEMENT_WEIGHTS, 'Complementary'
    else:
        w, label = cfg.SIMILAR_WEIGHTS, 'Similar'

    relation_weights = {
        'Visual (CLIP)': w['clip'],
        'GNN Graph':     w['gnn'],
        'Demographic':   w['demo'],
        'Category':      w['label'],
    }
    return top_segments, len(similar_idxs), relation_weights, label


def plot_graph_explanation(top_segments, total_similar, relation_weights, scoring_label):
    """Two-panel figure: demographic graph path (left) + relation-weight pie (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    # Left – who wore this item in the graph
    if top_segments:
        segs   = [s[:32] for s in top_segments.keys()]
        counts = list(top_segments.values())
        ax1.barh(segs, counts, color='#6C63FF')
        ax1.set_xlabel('Times worn', fontsize=8)
        ax1.set_title(f'Graph Path: Similar Users Who Wore This\n'
                      f'(matched {total_similar} training profiles)',
                      fontsize=9, fontweight='bold')
        ax1.invert_yaxis()
        ax1.tick_params(labelsize=7)
    else:
        ax1.text(0.5, 0.5, 'No direct graph path\nfound for this profile',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=9)
        ax1.set_title('Graph Path: Demographic Influence', fontsize=9)

    # Right – relation-type importance
    labels  = list(relation_weights.keys())
    sizes   = list(relation_weights.values())
    palette = ['#48CAE4', '#6C63FF', '#F77F00', '#2EC4B6']
    ax2.pie(sizes, labels=labels, colors=palette, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 8})
    ax2.set_title(f'Edge-Type Weights\n({scoring_label} scoring)',
                  fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig


def perform_virtual_try_on(person_image, garment_image):
    """Uses Gemini's native image generation (Nano Banana) for virtual try-on."""
    try:
        client = genai_new.Client(api_key=Config.GEMINI_API_KEY)

        # Convert PIL images to bytes for the new SDK
        def pil_to_bytes(img):
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()

        person_bytes = pil_to_bytes(person_image)
        garment_bytes = pil_to_bytes(garment_image)

        prompt = (
            """Virtual try-on: The first image shows a person. The second image shows 
            a clothing garment. Generate a single photorealistic image of the same 
            person wearing that garment. Preserve the person's face, body proportions, 
            pose, skin tone, hair, and background exactly. Only replace their clothing 
            with the garment from the second image. If the second image contains a dress, then remove the whole
            dress from the person image and replace it with the garment from the second image. And when replacing a top remove the existing top and replace it with the second image.
            """
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash-image',
            contents=[
                genai_types.Part.from_bytes(data=person_bytes, mime_type='image/jpeg'),
                genai_types.Part.from_bytes(data=garment_bytes, mime_type='image/jpeg'),
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                response_modalities=['IMAGE'],
            ),
        )

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    result_img = Image.open(io.BytesIO(part.inline_data.data))
                    return result_img, None

        return None, "No image was generated. Try again or use a clearer person photo."

    except Exception as e:
        return None, f"Try-on error: {str(e)}"

import base64

# Load background images
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# App-wide wallpaper
wallpaper_image = get_base64_image("E:/4 year/IRP/FYP/wallpaper 2.jpg")

# Streamlit UI
st.set_page_config(
    page_title="StyleSync AI - Outfit Recommender",
    page_icon="shirt",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS from external file
STYLE_PATH = os.path.join(os.path.dirname(__file__), 'style.css')
if os.path.exists(STYLE_PATH):
    with open(STYLE_PATH, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set dynamic CSS variables
st.markdown(f"""
<style>
:root {{
    --app-bg: url("data:image/jpg;base64,{wallpaper_image}");
}}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### User Details")
    gender = st.selectbox("Gender ", ["female", "male"], index=0)
    age_group = st.selectbox("Age Group ",
                             ["16-20", "21-25", "26-30", "31-40"], index=1)
    occasion = st.selectbox("Occasion ",
                            ["Party", "Prom", "Wedding", "Office",
                             "Dating", "Travel", "Sports"], index=0)

    st.markdown("---")
    st.markdown("### Upload Your Style")

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and len(uploaded_files) > 5:
        st.warning("Maximum 5 images allowed. Only the first 5 will be used.")
        uploaded_files = uploaded_files[:5]

    st.markdown("---")
    recommend_btn = st.button("Generate Recommendations",
                               type="primary", use_container_width=True)



st.markdown("""
<div class="hero-header">
    <div style="text-align:center">
        <div class="hero-title">StyleSync AI</div>
        <div class="hero-subtitle">
            Your personal AI stylist — find outfits you'll love
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

sys_data = load_system()

# Handle state for recommendations
if 'results' not in st.session_state:
    st.session_state.results = None
if 'pil_images' not in st.session_state:
    st.session_state.pil_images = []

if recommend_btn:
    if not uploaded_files:
        st.error("📸 Please upload at least one outfit photo before getting recommendations. "
                 "We need to see your style to find the best matches for you!")
    else:
        st.session_state.pil_images = []
        for uf in uploaded_files:
            img = Image.open(uf).convert('RGB')
            st.session_state.pil_images.append(img)

        with st.spinner("Analysing your style and finding the perfect outfits for you..."):
            st.session_state.results = run_recommendation(
                sys_data, st.session_state.pil_images,
                age_group, gender, occasion)

if st.session_state.results:
    results = st.session_state.results
    pil_images = st.session_state.pil_images

    if pil_images:
        st.markdown("### Your Style Photos")
        num_imgs = len(pil_images)
        cols = st.columns(min(num_imgs, 5))
        for i, img in enumerate(pil_images):
            with cols[i]:
                st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
                st.image(img, width=350, caption=f"Photo {i+1}")
                st.markdown('</div>', unsafe_allow_html=True)
                # st.image(img, width=350, caption=f"Photo {i+1}")

        if len(pil_images) > 1:
            st.markdown(
                f'<div class="multi-img-banner">'
                f'📸 <strong>{len(pil_images)} style photos analysed</strong> — '
                f'We blended your preferences across all photos to find outfits that suit your overall look.'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    def render_recommendations(title, items, icon="✨"):
        if not items:
            st.markdown(
                f'<div class="section-title">{icon} {title}</div>',
                unsafe_allow_html=True,
            )
            st.info("No matching items found for your profile. Try adjusting your preferences.")
            return

        n = len(items)
        if n < 5:
            st.warning(
                f"Only {n} matching items found. Try a different occasion or upload more style photos for better results."
            )

        st.markdown(
            f'<div class="section-title">{icon} {title} <span style="font-size:0.9rem;font-weight:400;color:#6b7280;">({n} items)</span></div>',
            unsafe_allow_html=True,
        )

        COLS_PER_ROW = 5
        tryon_cats = {'top', 'pants', 'skirt', 'dress', 'outer', 'rompers', 'leggings'}

        for row_start in range(0, n, COLS_PER_ROW):
            row_items = items[row_start:row_start + COLS_PER_ROW]
            # Pad with None so every row has COLS_PER_ROW columns
            padded = row_items + [None] * (COLS_PER_ROW - len(row_items))
            cols = st.columns(COLS_PER_ROW)

            for col_i, item in enumerate(padded):
                with cols[col_i]:
                    if item is None:
                        continue

                    global_idx = row_start + col_i
                    item_img = get_item_image(item['idx'], sys_data)

                    if item_img is not None:
                        st.image(item_img, use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="height:160px;background:#f3e8ff;border-radius:10px;'
                            'display:flex;align-items:center;justify-content:center;'
                            'color:#9ca3af;font-size:0.8rem;">No image</div>',
                            unsafe_allow_html=True,
                        )

                    true_cat_display = item['true_cat'].replace('_', ' ').title()
                    score_pct = max(0, min(100, item['score'] * 100))

                    st.markdown(
                        f'<div style="text-align:center;margin-top:0.3rem;">'
                        f'<span class="category-tag">{true_cat_display}</span>'
                        f'<span class="score-badge">{score_pct:.0f}% match</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    short_name = (
                        item['image_name'].split('-id_')[0]
                        if '-id_' in item['image_name']
                        else item['image_name']
                    )
                    st.caption(short_name.replace('WOMEN-', '').replace('MEN-', ''))

                    # Why this pick — simplified XAI
                    if st.button("✨ Why this pick?", key=f"exp_{title}_{global_idx}"):
                        with st.spinner("Getting style insight..."):
                            explanation = generate_outfit_explanation(
                                age_group, gender, occasion, item
                            )
                            st.markdown(f'<div class="xai-box">{explanation}</div>', unsafe_allow_html=True)

                    # Virtual Try-On
                    if item['true_cat'] in tryon_cats and pil_images and item_img is not None:
                        if st.button("👗 Try it on", key=f"vto_{title}_{global_idx}"):
                            with st.spinner("Creating your virtual try-on — this may take a moment..."):
                                tryon_img, err = perform_virtual_try_on(pil_images[0], item_img)
                                if tryon_img:
                                    st.image(tryon_img, caption="How it could look on you", use_container_width=True)
                                    st.success("Looks great on you!")
                                else:
                                    st.error("Could not generate try-on right now. Please try again.")

            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    render_recommendations("Outfit Recommendations For You", results['similar'], icon="✨")

    st.markdown("---")
    st.caption(
        f"Results for: {gender.title()} · {age_group} · {occasion} · "
        f"Primary garment: {results['primary_category'].replace('_',' ').title()}"
    )

else:
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="instruction-card">
        <h4>🧭 How It Works</h4>
        <ol>
            <li>Pick your <b>gender</b>, <b>age group</b> and <b>occasion</b> in the sidebar</li>
            <li>Upload <b>1–5 photos</b> of outfits you already love</li>
            <li>Hit <b>Get Recommendations</b></li>
            <li>Browse up to 10 personalised outfit picks</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="instruction-card">
        <h4>✨ What You Can Do</h4>
        <ul>
            <li>Tap <b>"Why this pick?"</b> to understand why an item was chosen</li>
            <li>Tap <b>"Try it on"</b> to see the item on you (works for tops, dresses, pants & more)</li>
            <li>Upload new photos anytime to refresh your recommendations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
