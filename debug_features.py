"""
Debug script to verify feature columns and model compatibility
"""
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DEBUGGING FEATURE COLUMNS MISMATCH")
print("=" * 80)

# Load data
df = pd.read_csv('cleaned_agriculture.csv')
print(f"\n✓ DataFrame loaded: {df.shape}")
print(f"  Columns in CSV: {list(df.columns)}")

# Load model
dt_model = joblib.load('final_dt_model.pkl')
print(f"\n✓ Model loaded: {type(dt_model)}")

# Check if model has feature_names_in_
if hasattr(dt_model, 'feature_names_in_'):
    model_features = dt_model.feature_names_in_.tolist()
    print(f"\n✓ Model's expected features (feature_names_in_): {len(model_features)} features")
    print(f"  Features: {model_features}")
else:
    print("\n✗ Model does NOT have feature_names_in_ attribute!")

# Load feature_columns pickle
try:
    feature_columns = joblib.load('feature_columns.pkl')
    print(f"\n✓ feature_columns.pkl loaded: {len(feature_columns)} features")
    print(f"  Features: {feature_columns}")
except Exception as e:
    print(f"\n✗ Error loading feature_columns.pkl: {e}")
    feature_columns = None

# Compare
if hasattr(dt_model, 'feature_names_in_'):
    model_features_set = set(model_features)
    csv_columns_set = set(df.columns)
    
    print(f"\n{'='*80}")
    print("COMPARISON:")
    print(f"{'='*80}")
    
    # Check if CSV columns match model features
    missing_in_csv = model_features_set - csv_columns_set
    extra_in_csv = csv_columns_set - model_features_set - {'Yield_Category'}
    
    if missing_in_csv:
        print(f"\n✗ Missing in CSV (model expects but not in data):")
        for feat in missing_in_csv:
            print(f"  - {feat}")
    
    if extra_in_csv:
        print(f"\n⚠ Extra in CSV (in data but model doesn't expect):")
        for feat in extra_in_csv:
            print(f"  - {feat}")
    
    if not missing_in_csv and not extra_in_csv:
        print(f"\n✓ All features match between model and CSV!")
    
    # Check dtypes
    print(f"\n{'='*80}")
    print("DATA TYPES:")
    print(f"{'='*80}")
    bool_cols = [col for col in model_features if col.startswith(('Crop_Type_', 'Adaptation_Strategies_'))]
    numeric_cols = [col for col in model_features if col not in bool_cols]
    
    print(f"\nBoolean columns ({len(bool_cols)}):")
    for col in bool_cols[:5]:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype} - sample: {df[col].iloc[0]}")
    
    print(f"\nNumeric columns ({len(numeric_cols)}):")
    for col in numeric_cols[:3]:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype} - sample: {df[col].iloc[0]}")

print(f"\n{'='*80}")
print("RECOMMENDATIONS:")
print(f"{'='*80}")
print("""
1. If model has feature_names_in_, use it directly (not pickle file)
2. Convert boolean string columns to actual bool type
3. Ensure feature order matches model's training order
4. Verify all features exist in the DataFrame
""")
