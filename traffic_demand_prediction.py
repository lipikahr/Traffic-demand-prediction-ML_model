# ============================================================
# Traffic Demand Prediction - Gridlock Hackathon
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

# ============================================================
# STEP 2: FEATURE ENGINEERING FUNCTION
# (applied identically to both train and test)
# ============================================================

def engineer_features(df):
    df = df.copy()

    # --- Timestamp: "8:30" → hour=8, minute=30 ---
    df['hour']   = df['timestamp'].apply(lambda x: int(str(x).split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[1]))

    # Cyclical encoding of hour (so hour 23 and hour 0 are treated as close)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Cyclical encoding of minute
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Peak hour flag: morning (7-9am) and evening (5-8pm) rush hours
    df['is_peak'] = df['hour'].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)

    # Night flag: low demand hours
    df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)

    # --- Day features ---
    df['day_of_week'] = df['day'] % 7
    df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
    df['week_number'] = df['day'] // 7

    # --- Geohash: decode to lat/lon ---
    # geohash strings like "qp02z1" encode a geographic bounding box
    # We approximate lat/lon from the first characters (coarse location)
    # Install geohash2 if needed: pip install geohash2
    try:
        import geohash2
        df['lat'] = df['geohash'].apply(lambda x: float(geohash2.decode(str(x))[0]))
        df['lon'] = df['geohash'].apply(lambda x: float(geohash2.decode(str(x))[1]))
        print("✓ Geohash decoded to lat/lon")
    except ImportError:
        print("⚠ geohash2 not installed. Run: pip install geohash2")
        print("  Using geohash prefix as fallback...")
        df['lat'] = 0.0
        df['lon'] = 0.0

    # --- Binary encode LargeVehicles and Landmarks ---
    df['LargeVehicles'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks']     = (df['Landmarks'] == 'Yes').astype(int)

    return df


train = engineer_features(train)
test  = engineer_features(test)


# ============================================================
# STEP 3: FILL MISSING VALUES
# ============================================================

# Temperature: fill with median per Weather type (logical grouping)
temp_median = train.groupby('Weather')['Temperature'].median()
for weather_type in temp_median.index:
    mask_train = (train['Weather'] == weather_type) & (train['Temperature'].isna())
    mask_test  = (test['Weather']  == weather_type) & (test['Temperature'].isna())
    train.loc[mask_train, 'Temperature'] = temp_median[weather_type]
    test.loc[mask_test,   'Temperature'] = temp_median[weather_type]

# Any remaining Temperature nulls → fill with overall median
overall_temp_median = train['Temperature'].median()
train['Temperature'] = train['Temperature'].fillna(overall_temp_median)
test['Temperature']  = test['Temperature'].fillna(overall_temp_median)

# RoadType: fill with most common value
most_common_road = train['RoadType'].mode()[0]
train['RoadType'] = train['RoadType'].fillna(most_common_road)
test['RoadType']  = test['RoadType'].fillna(most_common_road)

# Weather: fill with most common value
most_common_weather = train['Weather'].mode()[0]
train['Weather'] = train['Weather'].fillna(most_common_weather)
test['Weather']  = test['Weather'].fillna(most_common_weather)

print("✓ Missing values filled")


# ============================================================
# STEP 4: TARGET ENCODING
# (using mean demand per group — very powerful features)
# ============================================================

# Mean demand per geohash location
geo_mean = train.groupby('geohash')['demand'].mean().rename('geo_mean_demand')
train = train.join(geo_mean, on='geohash')
test  = test.join(geo_mean,  on='geohash')

# Mean demand per (geohash, hour) combination
geo_hour_mean = train.groupby(['geohash', 'hour'])['demand'].mean().rename('geo_hour_demand')
train = train.join(geo_hour_mean, on=['geohash', 'hour'])
test  = test.join(geo_hour_mean,  on=['geohash', 'hour'])

# Mean demand per (geohash, day_of_week) combination
geo_day_mean = train.groupby(['geohash', 'day_of_week'])['demand'].mean().rename('geo_day_demand')
train = train.join(geo_day_mean, on=['geohash', 'day_of_week'])
test  = test.join(geo_day_mean,  on=['geohash', 'day_of_week'])

# Mean demand per (RoadType, hour) combination
road_hour_mean = train.groupby(['RoadType', 'hour'])['demand'].mean().rename('road_hour_demand')
train = train.join(road_hour_mean, on=['RoadType', 'hour'])
test  = test.join(road_hour_mean,  on=['RoadType', 'hour'])

# Fill any NaN in target-encoded columns
# (happens when a geohash in test was never in train)
for col in ['geo_mean_demand', 'geo_hour_demand', 'geo_day_demand', 'road_hour_demand']:
    global_mean = train['demand'].mean()
    train[col] = train[col].fillna(global_mean)
    test[col]  = test[col].fillna(global_mean)

print("✓ Target encoding done")


# ============================================================
# STEP 5: LABEL ENCODE CATEGORICAL COLUMNS
# ============================================================

cat_cols = ['geohash', 'RoadType', 'Weather']

for col in cat_cols:
    le = LabelEncoder()
    # Fit on combined train+test to handle all categories
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

print("✓ Label encoding done")


# ============================================================
# STEP 6: DEFINE FEATURES
# ============================================================

features = [
    # Location
    'geohash', 'lat', 'lon',
    # Time
    'hour', 'minute', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'is_peak', 'is_night',
    # Day
    'day', 'day_of_week', 'is_weekend', 'week_number',
    # Road info
    'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    # Weather
    'Temperature', 'Weather',
    # Target-encoded features (most important!)
    'geo_mean_demand', 'geo_hour_demand', 'geo_day_demand', 'road_hour_demand'
]

X = train[features]
y = train['demand']
X_test = test[features]

print(f"✓ Feature matrix: {X.shape}")


# ============================================================
# STEP 7: LOG TRANSFORM TARGET
# demand is between 0 and 1 but right-skewed
# log1p compresses large values → better model performance
# ============================================================

y_log = np.log1p(y)


# ============================================================
# STEP 8: TRAIN WITH LIGHTGBM + 5-FOLD CROSS VALIDATION
# ============================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds  = np.zeros(len(train))   # out-of-fold predictions on train
test_preds = np.zeros(len(test))    # final predictions on test

lgb_params = {
    'n_estimators':     3000,
    'learning_rate':    0.03,
    'num_leaves':       127,
    'max_depth':        -1,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       0.1,
    'min_child_samples': 20,
    'random_state':     42,
    'n_jobs':           -1,
    'verbose':          -1
}

print("\n--- Training LightGBM with 5-Fold CV ---\n")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(500)
        ]
    )

    # Predict and reverse log transform (expm1 is inverse of log1p)
    val_preds = np.expm1(model.predict(X_val))
    oof_preds[val_idx] = val_preds

    test_preds += np.expm1(model.predict(X_test)) / 5

    fold_r2 = r2_score(y.iloc[val_idx], val_preds)
    print(f"Fold {fold+1} R²: {fold_r2:.4f}  |  Best iteration: {model.best_iteration_}")

overall_r2 = r2_score(y, oof_preds)
print(f"\n✓ Overall OOF R²: {overall_r2:.4f}")
print(f"  Competition score (approx): {max(0, 100 * overall_r2):.2f}")


# ============================================================
# STEP 9: FEATURE IMPORTANCE (helpful to understand the model)
# ============================================================

importance_df = pd.DataFrame({
    'feature':   features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 10 Most Important Features ---")
print(importance_df.head(10).to_string(index=False))


# ============================================================
# STEP 10: GENERATE SUBMISSION FILE
# ============================================================

submission = pd.DataFrame({
    'Index':  test['Index'],
    'demand': test_preds
})

submission.to_csv('dataset/submission.csv', index=False)

print(f"\n✓ submission.csv saved!")
print(f"  Shape: {submission.shape}  ← must be (41778, 2)")
print(f"  Preview:\n{submission.head()}")
