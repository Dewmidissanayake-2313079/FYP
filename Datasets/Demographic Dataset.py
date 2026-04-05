import pandas as pd
import numpy as np
from scipy import stats
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


SURVEY_PATH  = 'E:/4 year/IRP/FYP/Datasets/Demographic-Aware Personalized Outfit Recommendation  (Responses) - Form responses 1.csv'
DATASET_PATH = 'E:/4 year/IRP/FYP/Datasets/dataset.csv'
OUTPUT_PATH  = 'dataset_with_age_survey_based.csv'

AGE_GROUPS = {
    '16-20': (16, 20),
    '21-25': (21, 25),
    '26-30': (26, 30),
    '31-40': (31, 40)
}

# Occasion-based age boosts
OCCASION_BOOSTS = {
    'Prom':    {'16-20': 0.25},
    'Sports':  {'16-20': 0.15, '21-25': 0.10},
    'Party':   {'16-20': 0.10, '21-25': 0.10},
    'Dating':  {'21-25': 0.10, '26-30': 0.10},
    'Office':  {'26-30': 0.15, '31-40': 0.15},
    'Wedding': {'26-30': 0.10, '31-40': 0.10},
    'Travel':  {}
}

#  Balancing configuration 
MIN_VIABLE_GROUP_SIZE = 10

BALANCE_TARGET = 50

# Maximum samples per group
MAX_SAMPLES_PER_GROUP = 100

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
RULE_BASED_RATIO = 0.80


#  Survey statistics
def extract_survey_statistics(survey_df):

    survey_df.columns = [
        'timestamp', 'age_group', 'gender', 'occasions', 'comfort',
        'upper_body', 'lower_body', 'sleeve_length', 'lower_body_length',
        'color_palette', 'trend_follow', 'age_affects_fashion',
        'age_importance_system', 'willing_to_share', 'system_usefulness', 'email'
    ]

    survey_stats = {}

    for age_group in AGE_GROUPS.keys():
        age_data = survey_df[survey_df['age_group'] == age_group]
        n = len(age_data)
        if n == 0:
            continue

        print(f"\n  Age Group: {age_group} (n={n})")
        survey_stats[age_group] = {}

        sleeve_counts = age_data['sleeve_length'].str.split(',').explode().str.strip().value_counts(normalize=True)
        survey_stats[age_group]['sleeve'] = sleeve_counts.to_dict()

        lower_counts = age_data['lower_body'].str.split(',').explode().str.strip().value_counts(normalize=True)
        survey_stats[age_group]['lower_body'] = lower_counts.to_dict()

        length_counts = age_data['lower_body_length'].str.split(',').explode().str.strip().value_counts(normalize=True)
        survey_stats[age_group]['length'] = length_counts.to_dict()

        color_counts = age_data['color_palette'].str.split(',').explode().str.strip().value_counts(normalize=True)
        survey_stats[age_group]['color'] = color_counts.to_dict()

        print(f"    Top sleeve preference: {max(survey_stats[age_group]['sleeve'], key=survey_stats[age_group]['sleeve'].get)}")

    return survey_stats


#  Attribute mapping 
def create_attribute_mapping():
    return {
        'sleeve': {
            'Short':        ['short'],
            'Long sleeves': ['long', 'three_quarter'],
            'Sleeveless':   ['sleeveless'],
        },
        'lower_body': {
            'Trousers': ['pants/trousers'],
            'Jeans':    ['jeans'],
            'Shorts':   ['shorts'],
            'Skirts':   ['skirt'],
            'Leggings': ['leggings'],
        },
        'length': {
            'Full-Length':  ['full_length', 'calf_length'],
            'Knee-Length':  ['knee_length', 'thigh_length'],
            'Short':        ['extra_short'],
        },
        'color': {
            'Neutral': ['white', 'grey', 'beige', 'black', 'brown', 'navy'],
            'Dark':    ['black', 'navy', 'brown', 'purple'],
            'Bright':  ['red', 'pink', 'yellow', 'orange', 'green', 'blue'],
            'Mixed':   ['multi'],
        }
    }


def score_outfit_against_age_group(row, age_group, survey_stats, mapping):
    score   = 0.0
    matches = 0
    age_stats = survey_stats.get(age_group, {})
    if not age_stats:
        return 0.0

    sleeve_val = row.get('sleeve_length')
    if pd.notna(sleeve_val):
        for survey_key, dataset_vals in mapping['sleeve'].items():
            if sleeve_val in dataset_vals:
                score   += age_stats.get('sleeve', {}).get(survey_key, 0.0)
                matches += 1
                break

    lower_val = row.get('lower_body')
    if pd.notna(lower_val):
        for survey_key, dataset_vals in mapping['lower_body'].items():
            if lower_val in dataset_vals:
                score   += age_stats.get('lower_body', {}).get(survey_key, 0.0)
                matches += 1
                break

    length_val = row.get('lower_body_length')
    if pd.notna(length_val):
        for survey_key, dataset_vals in mapping['length'].items():
            if length_val in dataset_vals:
                score   += age_stats.get('length', {}).get(survey_key, 0.0)
                matches += 1
                break

    color_val = row.get('colour_top')
    if pd.notna(color_val):
        for survey_key, dataset_vals in mapping['color'].items():
            if str(color_val).lower() in dataset_vals:
                score   += age_stats.get('color', {}).get(survey_key, 0.0)
                matches += 1
                break

    if matches > 0:
        score /= matches

    occasion = row.get('occasion')
    if occasion in OCCASION_BOOSTS:
        score += OCCASION_BOOSTS[occasion].get(age_group, 0.0)

    return score


#  Age assignment
def assign_ages_rule_based(dataset_df, survey_stats, mapping):
    dataset_df       = dataset_df.copy()
    age_groups_list  = []
    ages_list        = []
    scores_by_group  = {ag: [] for ag in AGE_GROUPS.keys()}

    for idx, row in dataset_df.iterrows():
        scores = {ag: score_outfit_against_age_group(row, ag, survey_stats, mapping)
                  for ag in AGE_GROUPS.keys()}
        for ag, s in scores.items():
            scores_by_group[ag].append(s)

        best_ag = max(scores, key=scores.get)
        lo, hi  = AGE_GROUPS[best_ag]
        age_groups_list.append(best_ag)
        ages_list.append(np.random.randint(lo, hi + 1))

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(dataset_df)} rows...")

    dataset_df['age_group'] = age_groups_list
    dataset_df['age']       = ages_list

    print(f"\n  Age assignment complete!")
    print(f"\n  Resulting age distribution (based on dataset composition):")
    age_dist = dataset_df['age_group'].value_counts().sort_index()
    for ag, count in age_dist.items():
        print(f"    {ag}: {count:5d} ({count/len(dataset_df)*100:5.1f}%)")

    print(f"\n  Average compatibility scores by age group:")
    for ag in AGE_GROUPS.keys():
        print(f"    {ag}: {np.mean(scores_by_group[ag]):.3f}")

    return dataset_df


#  K-Modes clustering 
def perform_kmodes_clustering(dataset_df):
    categorical_cols = ['gender', 'occasion', 'upper_body', 'lower_body',
                        'sleeve_length', 'colour_top', 'lower_body_length']
    existing_cols = [c for c in categorical_cols if c in dataset_df.columns]

    cluster_data = dataset_df[existing_cols].fillna('unknown')
    print(f"\n  Clustering on {len(existing_cols)} attributes...")

    km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=0, random_state=RANDOM_SEED)
    dataset_df['cluster'] = km.fit_predict(cluster_data)

    crosstab = pd.crosstab(dataset_df['cluster'], dataset_df['age_group'], normalize='index')
    print(f"\n  Cluster → Age Group composition:")
    print((crosstab * 100).round(1))
    print(f"\n  Cluster purity: {crosstab.max(axis=1).mean():.3f}")

    return dataset_df


#  80/20 validation split 
def apply_80_20_split(dataset_df):
    n_total  = len(dataset_df)
    n_rule   = int(n_total * RULE_BASED_RATIO)
    n_random = n_total - n_rule

    random_indices = np.random.choice(dataset_df.index.tolist(), size=n_random, replace=False)
    dataset_df['validation_type'] = 'rule_based'
    dataset_df.loc[random_indices, 'validation_type'] = 'random_check'

    for idx in random_indices:
        lo, hi = AGE_GROUPS[dataset_df.loc[idx, 'age_group']]
        dataset_df.loc[idx, 'age'] = np.random.randint(lo, hi + 1)

    print(f"\n  Rule-based (80%): {n_rule}")
    print(f"  Random check (20%): {n_random}")
    return dataset_df


# Dataset Validation 
def validate_methodology(dataset_df, survey_df, survey_stats, mapping):
    short_sleeve = dataset_df[dataset_df['sleeve_length'] == 'short']
    if len(short_sleeve) > 0:
        dist = short_sleeve['age_group'].value_counts(normalize=True).sort_index()
        print(f"\n  Short sleeve outfits → Age distribution:")
        for ag, pct in dist.items():
            survey_pref = survey_stats.get(ag, {}).get('sleeve', {}).get('Short', 0.0)
            print(f"    {ag}: {pct*100:5.1f}% | Survey preference: {survey_pref*100:5.1f}%")
        young_pct = dist.get('16-20', 0) + dist.get('21-25', 0)
        tag = "PASS" if young_pct > 0.50 else "NOTE"
        print(f"\n  {tag}: {young_pct*100:.1f}% to young ages - {'expected' if tag=='PASS' else 'dataset may lack casual wear'}")

    for occasion, expected_ag, threshold in [
        ('Prom',   '16-20', 0.40),
        ('Office', '26-30', 0.25),
        ('Sports', '16-20', 0.30),
    ]:
        sub = dataset_df[dataset_df['occasion'] == occasion]
        if len(sub) > 0:
            pct = (sub['age_group'] == expected_ag).mean()
            print(f"\n  {occasion} → {expected_ag}: {pct*100:.1f}% (expect >{threshold*100:.0f}%) {'PASS' if pct >= threshold else 'CHECK'}")

    print("\n" + "=" * 70)
    print("TEST 3: CLUSTER-AGE COHERENCE")
    print("=" * 70)
    purity = pd.crosstab(dataset_df['cluster'], dataset_df['age_group'],
                         normalize='index').max(axis=1).mean()
    print(f"\n  Cluster purity: {purity:.3f}")
    print(f"   {'ACCEPTABLE' if purity > 0.45 else 'LOW'}: Clusters {'show' if purity > 0.45 else 'show weak'} age-related patterns")

    print(f"\n  Dataset size: {len(dataset_df)}")
    print(f"  Age range: {dataset_df['age'].min()} - {dataset_df['age'].max()}")
    print(f"  Mean age: {dataset_df['age'].mean():.1f}")
    print(f"\n  Final age distribution:")
    for ag, count in dataset_df['age_group'].value_counts().sort_index().items():
        print(f"    {ag}: {count:5d} ({count/len(dataset_df)*100:5.1f}%)")


#  Dataset balancing 
def balance_dataset(df):

    group_counts = df.groupby(['gender', 'age_group', 'occasion']).size()

    # Identify and drop sparse groups 
    sparse_groups = group_counts[group_counts < MIN_VIABLE_GROUP_SIZE]
    viable_groups = group_counts[group_counts >= MIN_VIABLE_GROUP_SIZE]

    if len(sparse_groups) > 0:
        print(f"\n  Dropping {len(sparse_groups)} structurally sparse groups "
              f"(< {MIN_VIABLE_GROUP_SIZE} samples):")
        for (gender, age_group, occasion), count in sparse_groups.items():
            print(f"    {gender}/{age_group}/{occasion}: {count} sample(s) — excluded")

    # Filter df to only viable groups
    df_viable = df.groupby(['gender', 'age_group', 'occasion']).filter(
        lambda x: len(x) >= MIN_VIABLE_GROUP_SIZE
    )

    print(f"\n  Viable groups: {len(viable_groups)} / {len(group_counts)} total")
    print(f"  Rows after sparse-group removal: {len(df_viable)} / {len(df)}")

    # Determine target per group 
    if isinstance(BALANCE_TARGET, int):
        target = min(BALANCE_TARGET, MAX_SAMPLES_PER_GROUP)
    elif BALANCE_TARGET == 'median':
        target = int(min(viable_groups.median(), MAX_SAMPLES_PER_GROUP))
    elif BALANCE_TARGET == 'percentile':
        target = int(min(viable_groups.quantile(0.25), MAX_SAMPLES_PER_GROUP))
    else:
        raise ValueError(f"Unknown BALANCE_TARGET: {BALANCE_TARGET}")

    print(f"\n  Target samples per group : {target}")
    print(f"  Maximum cap per group    : {MAX_SAMPLES_PER_GROUP}")
    print(f"  (Groups with < {target} samples will be oversampled with replacement)")
    print(f"  (Groups with > {MAX_SAMPLES_PER_GROUP} samples will be undersampled)\n")

    # Resample each group to target 
    balanced_parts = []
    summary_rows = []

    for (gender, age_group, occasion), group_df in df_viable.groupby(['gender', 'age_group', 'occasion']):
        n_original = len(group_df)

        if n_original >= target:
            # Undersample (no replacement needed)
            resampled = group_df.sample(n=target, random_state=RANDOM_SEED, replace=False)
            action = "undersampled"
        else:
            # Oversample with replacement
            resampled = group_df.sample(n=target, random_state=RANDOM_SEED, replace=True)
            action = f"oversampled ({n_original}→{target})"

        balanced_parts.append(resampled)
        summary_rows.append({
            'group': f"{gender}/{age_group}/{occasion}",
            'original': n_original,
            'balanced': target,
            'action': action
        })

    balanced_df = (pd.concat(balanced_parts)
                     .sample(frac=1, random_state=RANDOM_SEED)
                     .reset_index(drop=True))

    # Summary of the report 
    summary_df = pd.DataFrame(summary_rows)
    oversampled = summary_df[summary_df['action'].str.startswith('oversampled')]
    undersampled = summary_df[summary_df['action'] == 'undersampled']

    print(f"  Groups oversampled  : {len(oversampled)}")
    if len(oversampled) > 0:
        print(oversampled[['group', 'original', 'balanced']].to_string(index=False))

    print(f"\n  Groups undersampled : {len(undersampled)}")

    print(f"\n  {'─'*50}")
    print(f"  Balanced dataset size : {len(balanced_df)}")
    print(f"\n  Gender distribution:")
    print(balanced_df['gender'].value_counts().to_string())
    print(f"\n  Age group distribution:")
    print(balanced_df['age_group'].value_counts().sort_index().to_string())
    print(f"\n  Occasion distribution:")
    print(balanced_df['occasion'].value_counts().to_string())

    # Check imbalance ratio
    final_counts = balanced_df.groupby(['gender', 'age_group', 'occasion']).size()
    imbalance_ratio = final_counts.max() / final_counts.min()
    print(f"\n  Imbalance ratio (max/min): {imbalance_ratio:.2f}x  "
          f"{'✓ Good' if imbalance_ratio <= 2.0 else '⚠ Some variance remains'}")

    return balanced_df


def main():
    print("\n  Loading data")
    survey_df  = pd.read_csv(SURVEY_PATH)
    dataset_df = pd.read_csv(DATASET_PATH)
    print(f"  Survey: {len(survey_df)} responses")
    print(f"  Dataset: {len(dataset_df)} records")

    survey_stats = extract_survey_statistics(survey_df)
    mapping      = create_attribute_mapping()
    dataset_df   = assign_ages_rule_based(dataset_df, survey_stats, mapping)
    dataset_df   = perform_kmodes_clustering(dataset_df)
    dataset_df   = apply_80_20_split(dataset_df)
    validate_methodology(dataset_df, survey_df, survey_stats, mapping)

    dataset_df = balance_dataset(dataset_df)

    print("\n  Saving output")
    dataset_df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved: {OUTPUT_PATH}")
    print(f"  Final shape: {dataset_df.shape}")


if __name__ == "__main__":
    main()