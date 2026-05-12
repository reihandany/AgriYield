"""
Verification script untuk memastikan semua dependencies dan models tersedia
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("=" * 50)
    print("Checking Python Version...")
    print(f"Python {sys.version}")
    if sys.version_info >= (3, 8):
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False

def check_packages():
    """Check required packages"""
    print("\n" + "=" * 50)
    print("Checking Required Packages...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'imblearn': 'imbalanced-learn',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} (not installed)")
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\n⚠️ Missing packages detected:")
        print(f"Run: pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✓ All packages installed")
        return True

def check_data_files():
    """Check if all required data files exist"""
    print("\n" + "=" * 50)
    print("Checking Data Files...")
    
    required_files = {
        'cleaned_agriculture.csv': 'Dataset',
        'naive_bayes_model.pkl': 'Naive Bayes Model',
        'dt_smote_model.pkl': 'Decision Tree SMOTE Model',
        'encoder.pkl': 'Label Encoder',
        'feature_columns.pkl': 'Feature Columns'
    }
    
    missing_files = []
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024  # Size in KB
            print(f"✓ {description:30s} ({file_size:.1f} KB)")
        else:
            print(f"✗ {description:30s} (missing)")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✓ All data files present")
        return True

def test_models():
    """Test if models can be loaded"""
    print("\n" + "=" * 50)
    print("Testing Model Loading...")
    
    try:
        import joblib
        import pandas as pd
        
        # Load dataset
        df = pd.read_csv('cleaned_agriculture.csv')
        print(f"✓ Dataset loaded ({len(df)} rows, {len(df.columns)} columns)")
        
        # Load models
        nb_model = joblib.load('naive_bayes_model.pkl')
        print("✓ Naive Bayes model loaded")
        
        dt_model = joblib.load('dt_smote_model.pkl')
        print("✓ Decision Tree SMOTE model loaded")
        
        encoder = joblib.load('encoder.pkl')
        print(f"✓ Encoder loaded (classes: {encoder.classes_})")
        
        feature_columns = joblib.load('feature_columns.pkl')
        print(f"✓ Feature columns loaded ({len(feature_columns)} features)")
        
        # Test prediction
        X = df.drop('Yield_Category', axis=1)
        sample = X.iloc[0:1]
        
        nb_pred = nb_model.predict(sample)
        print(f"✓ Naive Bayes prediction test: {nb_pred[0]}")
        
        dt_pred = dt_model.predict(sample)
        print(f"✓ Decision Tree prediction test: {dt_pred[0]}")
        
        print("\n✓ All model tests passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during model testing: {str(e)}")
        return False

def main():
    """Run all checks"""
    print("\n" + "=" * 50)
    print("  AGRICULTURAL DASHBOARD SETUP VERIFICATION")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_packages(),
        check_data_files(),
        test_models()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("\n✓ ALL CHECKS PASSED!")
        print("\nYou can now run the dashboard with:")
        print("  streamlit run app.py")
        print("\nOr use the batch file:")
        print("  run_dashboard.bat")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nTo install dependencies, run:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == '__main__':
    exit_code = main()
    print("\n" + "=" * 50)
    input("Press Enter to exit...")
    sys.exit(exit_code)
