import os
import re
import clip
import torch
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

YOLO_MODEL_PATH = 'E:/4 year/IRP/FYP/runs/yolov8m_fashion5/weights/best.pt'
IMAGE_DIR       = 'E:/4 year/IRP/FYP/DeepFashion Dataset/images'
OUTPUT_DIR      = 'E:/4 year/IRP/FYP/features'
CROPS_DIR       = os.path.join(OUTPUT_DIR, 'crops')
CONF_THRESHOLD  = 0.3

FASHION_CLASSES = {
    0: 'top', 1: 'outer', 2: 'skirt', 3: 'dress', 4: 'pants',
    5: 'leggings', 6: 'headwear', 7: 'eyeglass', 8: 'neckwear',
    9: 'belt', 10: 'footwear', 11: 'bag', 12: 'ring', 13: 'wrist_wearing',
    14: 'socks', 15: 'gloves', 16: 'necklace', 17: 'rompers',
    18: 'earrings', 19: 'tie'
}

os.makedirs(CROPS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

yolo_model = YOLO(YOLO_MODEL_PATH)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
print("Models loaded\n")


def assign_gender(name):
    if name.upper().startswith('MEN'):
        return 'male'
    elif name.upper().startswith('WOMEN'):
        return 'female'
    return 'unknown'


def extract_clip_features(crop_pil):
    image_tensor = clip_preprocess(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()


def detect_and_extract(image_path, save_crops=True):
    img_name    = Path(image_path).stem
    results_list = []
    yolo_results = yolo_model(image_path, conf=CONF_THRESHOLD, verbose=False)
    img_pil      = Image.open(image_path).convert('RGB')
    img_w, img_h = img_pil.size
    gender       = assign_gender(img_name)

    for result in yolo_results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf     = float(box.conf[0])
            cls_id   = int(box.cls[0])
            class_name = FASHION_CLASSES.get(cls_id, 'unknown')

            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            x1p = max(0, x1 - pad_x);     y1p = max(0, y1 - pad_y)
            x2p = min(img_w, x2 + pad_x); y2p = min(img_h, y2 + pad_y)

            crop = img_pil.crop((x1p, y1p, x2p, y2p))
            if crop.width < 20 or crop.height < 20:
                continue

            crop_filename = f"{img_name}_item{i}_{class_name}.jpg"
            crop_path     = os.path.join(CROPS_DIR, crop_filename)
            if save_crops:
                crop.save(crop_path)

            features = extract_clip_features(crop)
            results_list.append({
                'image_name': img_name,
                'item_id':    f"{img_name}_item{i}",
                'class_id':   cls_id,
                'class_name': class_name,
                'confidence': round(conf, 4),
                'gender':     gender,
                'bbox':       [x1, y1, x2, y2],
                'crop_path':  crop_path,
                'clip_features': features
            })
    return results_list


old_csv = os.path.join(OUTPUT_DIR, 'item_metadata.csv')
old_npy = os.path.join(OUTPUT_DIR, 'clip_features.npy')
old_ids = os.path.join(OUTPUT_DIR, 'item_ids.csv')

checkpoint_csv = os.path.join(OUTPUT_DIR, 'item_metadata_v2.csv')
features_npy   = os.path.join(OUTPUT_DIR, 'clip_features_v2.npy')
item_ids_csv   = os.path.join(OUTPUT_DIR, 'item_ids_v2.csv')

df_existing    = pd.read_csv(old_csv)
feats_existing = np.load(old_npy)

df_existing['gender']    = df_existing['image_name'].apply(assign_gender)
df_existing['crop_path'] = df_existing.apply(
    lambda row: os.path.join(CROPS_DIR, f"{row['item_id']}_{row['class_name']}.jpg"), axis=1
)
df_existing.to_csv(checkpoint_csv, index=False)
print(f"  Fixed gender: {df_existing['gender'].value_counts().to_dict()}")
print(f"  All {len(df_existing)} rows now have crop_path\n")


#  Select new images 
all_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
processed_images = set(df_existing['image_name'].unique())

remaining_by_cat = {}
for f in all_files:
    stem = os.path.splitext(f)[0]
    if stem in processed_images:
        continue
    m   = re.match(r'((?:MEN|WOMEN)-[^-]+)', f)
    cat = m.group(1) if m else 'other'
    remaining_by_cat.setdefault(cat, []).append(f)

TARGETS = {
    'WOMEN-Dresses':             1450,

    'WOMEN-Blouses_Shirts':       700,

    'WOMEN-Tees_Tanks':           650,

    'WOMEN-Skirts':               650,

    'WOMEN-Jackets_Coats':        500,

    'WOMEN-Pants':                450,

    'WOMEN-Sweatshirts_Hoodies':  400,

    'WOMEN-Sweaters':             400,

    'WOMEN-Shorts':               400,

    'WOMEN-Denim':                146,

    'WOMEN-Leggings':             100,

    'WOMEN-Rompers_Jumpsuits':    300,


    'WOMEN-Graphic_Tees':         150,

    #  MEN categories
    'MEN-Shirts_Polos':           644,

    'MEN-Pants':                  239,

    'MEN-Suiting':                 25,

    'MEN-Tees_Tanks':            2541,

    'MEN-Jackets_Vests':          399,

    'MEN-Shorts':                 228,

    'MEN-Denim':                   97,

    'MEN-Sweatshirts_Hoodies':    708,

    'MEN-Sweaters':               531,
}

#  Select and process images 
new_images = []
print("  Category selection:")
print(f"  {'Category':<30} {'Available':>10} {'Selected':>10}")
print("  " + "─" * 54)

total_w = 0
total_m = 0
for cat, max_count in TARGETS.items():
    available = remaining_by_cat.get(cat, [])
    selected  = available[:max_count]
    new_images.extend(selected)
    gender_tag = "women" if cat.startswith("WOMEN") else "men"
    if gender_tag == "women":
        total_w += len(selected)
    else:
        total_m += len(selected)
    status = "OK" if len(available) >= max_count else f"ONLY {len(available)} available"
    print(f"  {cat:<30} {len(available):>10} {len(selected):>10}  {status}")

total = total_w + total_m
print(f"\n  Women subtotal : {total_w:,}  ({total_w/total*100:.1f}%)")
print(f"  Men subtotal   : {total_m:,}  ({total_m/total*100:.1f}%)")
print(f"  Grand total    : {total:,} new images\n")


#  Extraction loop 
all_records    = df_existing.to_dict('records')
feature_matrix = list(feats_existing)
item_ids       = list(df_existing['item_id'])

for i, img_file in enumerate(tqdm(new_images, desc="Extracting")):
    img_path = os.path.join(IMAGE_DIR, img_file)
    try:
        items = detect_and_extract(img_path, save_crops=True)
        for item in items:
            feat = item.pop('clip_features')
            all_records.append(item)
            feature_matrix.append(feat)
            item_ids.append(item['item_id'])
    except Exception as e:
        print(f"\nError on {img_file}: {e}")
        continue

    if (i + 1) % 500 == 0:
        pd.DataFrame(all_records).to_csv(checkpoint_csv, index=False)
        np.save(features_npy, np.array(feature_matrix))
        pd.Series(item_ids).to_csv(item_ids_csv, index=False)
        print(f"\n  Checkpoint: {i+1}/{len(new_images)} done, {len(all_records)} total items")


#  Save final output 
df_final      = pd.DataFrame(all_records)
feature_array = np.array(feature_matrix)

df_final.to_csv(checkpoint_csv, index=False)
np.save(features_npy, feature_array)
pd.Series(item_ids).to_csv(item_ids_csv, index=False)

print(f"\n  Total items    : {len(df_final):,}")
print(f"  Feature shape  : {feature_array.shape}")
print(f"\n  Gender breakdown:")
print(df_final['gender'].value_counts().to_string())
print(f"\n  Detected class breakdown:")
print(df_final['class_name'].value_counts().to_string())


print("\n  Renaming _v2 files to replace originals...")
for src, dst in [
    (checkpoint_csv, old_csv),
    (features_npy,   old_npy),
    (item_ids_csv,   old_ids),
]:
    try:
        os.replace(src, dst)
        print(f"  {os.path.basename(src)} → {os.path.basename(dst)}")
    except PermissionError:
        print(f"  WARNING: Could not replace {os.path.basename(dst)} — file locked")
        print(f"  Manually rename: {os.path.basename(src)} → {os.path.basename(dst)}")