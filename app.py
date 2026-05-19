import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
st.set_page_config(
    page_title="AgriYield",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
navy_blue = "#001f3f"
light_navy = "#003d5c"
accent_gold = "#FFD700"
text_color = "#ffffff"

# Custom CSS untuk tema navy blue
st.markdown(f"""
    <style>
        :root {{
            --primary-color: {navy_blue};
            --secondary-color: {light_navy};
            --accent-color: {accent_gold};
            --text-color: {text_color};
        }}
        
        /* Main content area */
        .stMainBlockContainer {{
            background-color: #f8f9fa;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {navy_blue} !important;
        }}
        
        [data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] button {{
            background-color: {light_navy};
            color: {text_color};
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {accent_gold} !important;
            color: {navy_blue} !important;
        }}
        
        /* Metric card */
        .metric-card {{
            background: linear-gradient(135deg, {navy_blue} 0%, {light_navy} 100%);
            padding: 20px;
            border-radius: 10px;
            color: {text_color};
            border-left: 5px solid {accent_gold};
        }}
        
        /* Headings */
        h1 {{
            color: {navy_blue};
            border-bottom: 3px solid {accent_gold};
            padding-bottom: 10px;
        }}
        
        h2 {{
            color: {navy_blue};
        }}
        
        h3 {{
            color: {navy_blue};
        }}
        
        /* Subheader */
        [data-testid="stMarkdownContainer"] h4 {{
            color: {navy_blue};
        }}
        
        /* Buttons */
        .stButton > button {{
            background-color: {accent_gold};
            color: {navy_blue};
            font-weight: bold;
            border: none;
        }}
        
        .stButton > button:hover {{
            background-color: #FFC700;
            color: {navy_blue};
        }}
        
        /* Sliders */
        .stSlider > div > div > div > div {{
            color: {navy_blue};
        }}
        
        /* Selectbox */
        .stSelectbox {{
            color: {navy_blue};
        }}
        
        /* Input fields */
        input, select, textarea {{
            background-color: #ffffff !important;
            color: {navy_blue} !important;
            border: 2px solid {light_navy} !important;
        }}
        
        input:focus, select:focus, textarea:focus {{
            border: 2px solid {accent_gold} !important;
        }}
        
        /* Dataframe */
        [data-testid="stDataFrame"] {{
            background-color: transparent !important;
        }}
        [data-testid="stDataFrame"] table {{
            background: linear-gradient(135deg, {light_navy} 0%, {navy_blue} 100%) !important;
            color: {text_color} !important;
        }}
        [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {{
            color: {text_color} !important;
            border-color: rgba(255,255,255,0.08) !important;
        }}
        
        /* Metric boxes */
        [data-testid="metric-container"] {{
            background: linear-gradient(135deg, {light_navy} 0%, {navy_blue} 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid {accent_gold};
        }}
        
        /* Info/Warning/Success boxes */
        .stAlert {{
            background-color: #f0f2f6;
            border-radius: 5px;
        }}
    </style>
""", unsafe_allow_html=True)

# Helper to apply transparent/ themed layout to Plotly figures
def apply_plotly_theme(fig):
    try:
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=text_color)
        )
    except Exception:
        pass
    return fig

# ==================== LOAD DATA & MODELS ====================
@st.cache_resource
def load_data_and_models():
    df = pd.read_csv('cleaned_agriculture.csv')
    
    # Convert boolean columns properly (handle string "True"/"False")
    bool_cols = [col for col in df.columns if col.startswith(('Crop_Type_', 'Adaptation_Strategies_'))]
    for col in bool_cols:
        if df[col].dtype == 'object':
            # Convert string "True"/"False" to actual bool
            df[col] = df[col].map({'True': True, 'False': False})
        elif df[col].dtype != 'bool':
            # Convert other types to bool
            df[col] = df[col].astype(bool)
    
    dt_model = joblib.load('initial_dt_model.pkl')
    encoder = joblib.load('encoder.pkl')
    
    # Use model's feature_names_in_ if available, otherwise load from pickle
    if hasattr(dt_model, 'feature_names_in_'):
        feature_columns = dt_model.feature_names_in_.tolist()
    else:
        feature_columns = joblib.load('feature_columns.pkl')
    
    return df, dt_model, encoder, feature_columns

df, dt_model, encoder, feature_columns = load_data_and_models()

# ==================== UTILITY FUNCTIONS ====================
def get_crop_types():
    """Extract crop types from feature columns"""
    crop_features = [col for col in feature_columns if col.startswith('Crop_Type_')]
    return [col.replace('Crop_Type_', '') for col in crop_features]

def get_adaptation_strategies():
    """Extract adaptation strategies from feature columns"""
    adapt_features = [col for col in feature_columns if col.startswith('Adaptation_Strategies_')]
    return [col.replace('Adaptation_Strategies_', '') for col in adapt_features]

def create_input_features(temp, precip, co2, extreme, irrig, soil, fert, pest, crop, adaptation):
    """Create feature array for prediction"""
    features_dict = {}
    
    # Numeric features
    features_dict['Average_Temperature_C'] = temp
    features_dict['Total_Precipitation_mm'] = precip
    features_dict['CO2_Emissions_MT'] = co2
    features_dict['Extreme_Weather_Events'] = extreme
    features_dict['Irrigation_Access_%'] = irrig
    features_dict['Soil_Health_Index'] = soil
    features_dict['Fertilizer_Use_KG_per_HA'] = fert
    features_dict['Pesticide_Use_KG_per_HA'] = pest
    
    # Crop type one-hot encoding
    for crop_option in get_crop_types():
        features_dict[f'Crop_Type_{crop_option}'] = (crop == crop_option)
    
    # Adaptation strategies one-hot encoding
    for adapt_option in get_adaptation_strategies():
        features_dict[f'Adaptation_Strategies_{adapt_option}'] = (adaptation == adapt_option)
    
    # Create array with values in correct order, converting bools to proper type
    feature_values = []
    for col in feature_columns:
        val = features_dict.get(col, 0)
        # Ensure boolean columns are proper bool type
        if col.startswith(('Crop_Type_', 'Adaptation_Strategies_')):
            feature_values.append(bool(val))
        else:
            feature_values.append(val)
    
    feature_array = np.array([feature_values])
    return feature_array

def get_model_performance():
    """Calculate model performance metrics on test set (80-20 split)"""
    # Select only the features the model was trained on
    # (df already has boolean columns converted correctly in load_data_and_models)
    X = df[feature_columns]
    y = df['Yield_Category']
    
    # Split data 80-20 (train-test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Predictions on test set only
    dt_pred = dt_model.predict(X_test)
    
    metrics = {
        'Decision Tree': {
            'accuracy': accuracy_score(y_test, dt_pred),
            'precision': precision_score(y_test, dt_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, dt_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, dt_pred, average='weighted', zero_division=0),
            'predictions': dt_pred
        }
    }
    return metrics, y_test

# ==================== PAGE FUNCTIONS ====================
def page_home():
    """Home Page"""
    st.title("🌾 AgriYield - Dashboard Prediksi Produktivitas Pertanian")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #001f3f 0%, #003d5c 100%); padding: 30px; border-radius: 10px; color: white; margin-bottom: 30px;">
        <h3>Sistem Pendukung Keputusan Berbasis Perubahan Iklim</h3>
        <p>Dashboard ini dirancang untuk membantu menganalisis pengaruh faktor lingkungan terhadap tingkat produktivitas hasil panen. 
        Sistem mengintegrasikan visualisasi data interaktif dan model machine learning untuk memberikan gambaran kondisi pertanian berdasarkan variabel iklim dan praktik pengelolaan lahan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Data", f"{len(df):,}", help="Jumlah sampel dalam dataset")
    with col2:
        st.metric("📈 Total Fitur", f"{len(feature_columns)}", help="Jumlah variabel input")
    with col3:
        st.metric("🎯 Kelas Target", "2", help="rendah, tinggi")
    with col4:
        metrics, _ = get_model_performance()
        best_acc = metrics['Decision Tree']['accuracy']
        st.metric("🏆 Model Accuracy", f"{best_acc:.2%}", help="Akurasi Decision Tree Model")
    
    st.divider()
    
    # Key Features
    st.subheader("📋 Fitur Utama Dashboard")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔍 Analisis Data
        - **Dataset Overview**: Karakteristik data dan distribusi kelas
        - **EDA**: Visualisasi hubungan antar variabel
        - **Model Evaluation**: Evaluasi performa model
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 Machine Learning
        - **ML Prediction**: Simulasi prediksi real-time
        - **Batch Prediction**: Prediksi multiple data sekaligus
        - **Risk Assessment**: Analisis risiko produktivitas
        """)
    
    st.divider()
    
    # Model Information
    st.subheader("🧠 Model Machine Learning")
    st.markdown("""
    ### Decision Tree
    - Model klasifikasi berbasis pohon keputusan
    - Dilengkapi dengan teknik SMOTE untuk menangani data tidak seimbang
    - Interpretasi hasil yang mudah dipahami
    - Performa tinggi dengan akurasi optimal
    """)
    
    st.divider()
    st.info("💡 **Tips**: Gunakan ML Prediction untuk simulasi real-time atau Batch Prediction untuk prediksi multiple data dengan upload CSV!")

def page_dataset_overview():
    """Dataset Overview Page"""
    st.title("📊 Dataset Overview")
    
    # Dataset Info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sampel", len(df))
    with col2:
        st.metric("Total Fitur", len(feature_columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.divider()
    
    # Yield Category Distribution
    st.subheader("📈 Distribusi Kategori Produktivitas")
    yield_counts = df['Yield_Category'].value_counts()
    
    # Color mapping untuk setiap kategori
    color_map = {'rendah': '#FF6B6B', 'tinggi': '#45B7D1'}
    colors = [color_map.get(cat, '#999999') for cat in yield_counts.index]
    
    fig_yield = go.Figure(data=[
        go.Bar(
            x=yield_counts.index,
            y=yield_counts.values,
            marker=dict(color=colors),
            text=yield_counts.values,
            textposition='auto',
        )
    ])
    fig_yield.update_layout(
        title="Distribusi Kategori Produktivitas",
        xaxis_title="Kategori Produktivitas",
        yaxis_title="Jumlah Sampel",
        template="plotly_white",
        hovermode='x unified'
    )
    apply_plotly_theme(fig_yield)
    st.plotly_chart(fig_yield, use_container_width=True)
    
    # Feature Statistics
    st.subheader("📋 Statistik Deskriptif")
    numeric_cols = ['Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT',
                   'Extreme_Weather_Events', 'Irrigation_Access_%', 'Soil_Health_Index',
                   'Fertilizer_Use_KG_per_HA', 'Pesticide_Use_KG_per_HA']
    
    st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
    
    # Crop Type Distribution
    st.subheader("🌱 Jenis Tanaman")
    crop_cols = [col for col in feature_columns if col.startswith('Crop_Type_')]
    crop_counts = {}
    for col in crop_cols:
        crop_name = col.replace('Crop_Type_', '')
        crop_counts[crop_name] = df[col].sum()
    
    crop_df = pd.DataFrame(list(crop_counts.items()), columns=['Tanaman', 'Jumlah'])
    fig_crop = px.pie(crop_df, values='Jumlah', names='Tanaman', 
                     title="Distribusi Jenis Tanaman")
    apply_plotly_theme(fig_crop)
    st.plotly_chart(fig_crop, use_container_width=True)
    
    # Adaptation Strategies Distribution
    st.subheader("🌍 Strategi Adaptasi")
    adapt_cols = [col for col in feature_columns if col.startswith('Adaptation_Strategies_')]
    adapt_counts = {}
    for col in adapt_cols:
        adapt_name = col.replace('Adaptation_Strategies_', '')
        adapt_counts[adapt_name] = df[col].sum()
    
    adapt_df = pd.DataFrame(list(adapt_counts.items()), columns=['Strategi', 'Jumlah'])
    fig_adapt = px.bar(adapt_df, x='Strategi', y='Jumlah',
                      title="Distribusi Strategi Adaptasi",
                      labels={'Jumlah': 'Jumlah Sampel'})
    apply_plotly_theme(fig_adapt)
    st.plotly_chart(fig_adapt, use_container_width=True)

def page_eda():
    """EDA Page"""
    st.title("🔍 Exploratory Data Analysis")
    
    # Temperature Distribution
    st.subheader("🌡️ Distribusi Suhu Rata-rata")
    fig_temp = px.histogram(df, x='Average_Temperature_C', nbins=30,
                           title="Distribusi Suhu Rata-rata",
                           labels={'Average_Temperature_C': 'Suhu (°C)'})
    apply_plotly_theme(fig_temp)
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Precipitation Distribution
    st.subheader("🌧️ Distribusi Curah Hujan")
    fig_precip = px.histogram(df, x='Total_Precipitation_mm', nbins=30,
                             title="Distribusi Curah Hujan",
                             labels={'Total_Precipitation_mm': 'Curah Hujan (mm)'})
    apply_plotly_theme(fig_precip)
    st.plotly_chart(fig_precip, use_container_width=True)
    
    # Soil Health Distribution
    st.subheader("🌱 Distribusi Kesehatan Tanah")
    fig_soil = px.histogram(df, x='Soil_Health_Index', nbins=30,
                           title="Distribusi Indeks Kesehatan Tanah",
                           labels={'Soil_Health_Index': 'Soil Health Index'})
    apply_plotly_theme(fig_soil)
    st.plotly_chart(fig_soil, use_container_width=True)
    
    # Extreme Weather Events
    st.subheader("⛈️ Distribusi Peristiwa Cuaca Ekstrem")
    fig_extreme = px.histogram(df, x='Extreme_Weather_Events', nbins=30,
                              title="Distribusi Peristiwa Cuaca Ekstrem",
                              labels={'Extreme_Weather_Events': 'Kejadian Ekstrem'})
    apply_plotly_theme(fig_extreme)
    st.plotly_chart(fig_extreme, use_container_width=True)
    
    st.divider()
    
    # Relationships with Yield
    st.subheader("📊 Hubungan Variabel terhadap Produktivitas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temp_yield = px.scatter(df, x='Average_Temperature_C', y='Total_Precipitation_mm',
                                   color='Yield_Category',
                                   title="Suhu vs Curah Hujan (warna: Produktivitas)",
                                   color_discrete_map={'rendah': '#FF6B6B', 'tinggi': '#45B7D1'})
        apply_plotly_theme(fig_temp_yield)
        st.plotly_chart(fig_temp_yield, use_container_width=True)
    
    with col2:
        fig_soil_yield = px.box(df, x='Yield_Category', y='Soil_Health_Index',
                               title="Kesehatan Tanah per Kategori Produktivitas",
                               color='Yield_Category',
                               color_discrete_map={'rendah': '#FF6B6B', 'tinggi': '#45B7D1'})
        apply_plotly_theme(fig_soil_yield)
        st.plotly_chart(fig_soil_yield, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig_irrig_yield = px.box(df, x='Yield_Category', y='Irrigation_Access_%',
                                title="Akses Irigasi per Kategori Produktivitas",
                                color='Yield_Category',
                                color_discrete_map={'rendah': '#FF6B6B', 'tinggi': '#45B7D1'})
        apply_plotly_theme(fig_irrig_yield)
        st.plotly_chart(fig_irrig_yield, use_container_width=True)
    
    with col4:
        fig_co2_yield = px.box(df, x='Yield_Category', y='CO2_Emissions_MT',
                              title="Emisi CO₂ per Kategori Produktivitas",
                              color='Yield_Category',
                              color_discrete_map={'rendah': '#FF6B6B', 'tinggi': '#45B7D1'})
        apply_plotly_theme(fig_co2_yield)
        st.plotly_chart(fig_co2_yield, use_container_width=True)

def page_ml_prediction():
    """ML Prediction Page"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {navy_blue} 0%, {light_navy} 100%); padding: 30px; border-radius: 10px; color: white; margin-bottom: 30px;">
        <h2 style="color: {accent_gold}; margin: 0;">🤖 Machine Learning Prediction</h2>
        <p style="margin: 10px 0 0 0;">Simulasikan prediksi produktivitas pertanian dengan memasukkan parameter lingkungan</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Parameter Iklim")
        temp = st.slider("Suhu Rata-rata (°C)", -50.0, 50.0, 20.0, step=0.5)
        precip = st.slider("Curah Hujan (mm)", 0.0, 500.0, 100.0, step=1.0)
        co2 = st.slider("Emisi CO₂ (MT)", 0.0, 100.0, 50.0, step=1.0)
        extreme = st.slider("Peristiwa Cuaca Ekstrem", 0.0, 10.0, 2.0, step=0.1)
    
    with col2:
        st.subheader("🌱 Parameter Pengelolaan Lahan")
        irrig = st.slider("Akses Irigasi (%)", 0.0, 100.0, 50.0, step=1.0)
        soil = st.slider("Indeks Kesehatan Tanah", 0.0, 10.0, 5.0, step=0.1)
        fert = st.slider("Penggunaan Pupuk (KG/HA)", 0.0, 500.0, 200.0, step=1.0)
        pest = st.slider("Penggunaan Pestisida (KG/HA)", 0.0, 100.0, 20.0, step=0.5)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        crop = st.selectbox("Jenis Tanaman", get_crop_types())
    
    with col2:
        adaptation = st.selectbox("Strategi Adaptasi", get_adaptation_strategies())
    
    st.divider()
    
    if st.button("🔮 Prediksi Produktivitas", key="predict_btn", use_container_width=True):
        # Create features
        features = create_input_features(temp, precip, co2, extreme, irrig, soil, fert, pest, crop, adaptation)
        
        # Make predictions
        dt_pred = dt_model.predict(features)[0]
        dt_proba = dt_model.predict_proba(features)[0]
        
        # Display Results
        st.subheader("📊 Hasil Prediksi")
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #001f3f 0%, #003d5c 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;
                    border-left: 5px solid #FFD700;">
            <h4>Decision Tree</h4>
            <h2 style="color: #FFD700;">{dt_pred.upper()}</h2>
            <p>Probabilitas: {max(dt_proba):.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Probability Chart
        st.subheader("📊 Distribusi Probabilitas")
        
        # Get model classes dan filter probabilities sesuai encoder
        model_classes = dt_model.classes_
        encoder_indices = [list(model_classes).index(cat) for cat in encoder if cat in model_classes]
        filtered_categories = [encoder[i] for i, cat in enumerate(encoder) if cat in model_classes]
        
        filtered_proba_dt = dt_proba[encoder_indices]
        dt_proba_df = pd.DataFrame({
            'Kategori': filtered_categories,
            'Probabilitas': filtered_proba_dt
        })
        fig_dt = px.bar(dt_proba_df, x='Kategori', y='Probabilitas',
                       title="Probabilitas Decision Tree",
                       color='Kategori',
                       color_discrete_map={'rendah': '#FF6B6B', 'tinggi': '#45B7D1'})
        apply_plotly_theme(fig_dt)
        st.plotly_chart(fig_dt, use_container_width=True)
        
        st.divider()
        
        # Risk Assessment
        st.subheader("⚠️ Analisis Risiko")
        risk_level = "RENDAH" if dt_pred == "tinggi" else "TINGGI"
        
        risk_color = "#45B7D1" if risk_level == "RENDAH" else "#FF6B6B"
        
        st.markdown(f"""
        <div style="background: {risk_color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h3>Tingkat Risiko: {risk_level}</h3>
            <p>Berdasarkan prediksi Decision Tree</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("💡 Rekomendasi")
        if risk_level == "RENDAH":
            st.success("✅ Kondisi pertanian optimal. Lanjutkan praktik pengelolaan lahan yang sudah diterapkan.")
        else:
            st.warning("⚠️ Risiko produktivitas tinggi. Pertimbangkan:\n- Peningkatan akses irigasi\n- Perbaikan kesehatan tanah\n- Penerapan strategi adaptasi yang lebih baik")

def page_batch_prediction():
    """Batch Prediction Page"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {navy_blue} 0%, {light_navy} 100%); padding: 30px; border-radius: 10px; color: white; margin-bottom: 30px;">
        <h2 style="color: {accent_gold}; margin: 0;">📤 Batch Prediction</h2>
        <p style="margin: 10px 0 0 0;">Lakukan prediksi untuk multiple data sekaligus dengan upload file CSV</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for upload and instructions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📥 Upload File CSV")
        uploaded_file = st.file_uploader(
            "Pilih file CSV dengan data pertanian", 
            type=['csv'],
            help="File harus memiliki kolom yang sesuai dengan fitur model"
        )
    
    with col2:
        st.subheader("📋 Template")
        # Provide a sample template
        if st.button("📥 Download Template"):
            # Create template with first 5 rows from dataset
            template_df = df[feature_columns].head(5).copy()
            
            csv = template_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Template CSV",
                data=csv,
                file_name="template_batch_prediction.csv",
                mime="text/csv"
            )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            uploaded_df = pd.read_csv(uploaded_file)
            
            st.subheader(f"📊 Data yang Di-Upload ({len(uploaded_df)} baris)")
            st.dataframe(uploaded_df, use_container_width=True)
            
            # Validate columns
            missing_cols = set(feature_columns) - set(uploaded_df.columns)
            if missing_cols:
                st.error(f"❌ Kolom yang hilang: {', '.join(missing_cols)}")
                st.info(f"✅ Kolom yang diperlukan: {', '.join(feature_columns)}")
                return
            
            # Convert boolean columns
            bool_cols = [col for col in feature_columns if col.startswith(('Crop_Type_', 'Adaptation_Strategies_'))]
            uploaded_df_copy = uploaded_df[feature_columns].copy()
            for col in bool_cols:
                if uploaded_df_copy[col].dtype == 'object':
                    uploaded_df_copy[col] = uploaded_df_copy[col].map({'True': True, 'False': False})
                elif uploaded_df_copy[col].dtype != 'bool':
                    uploaded_df_copy[col] = uploaded_df_copy[col].astype(bool)
            
            # Make predictions
            st.subheader("🔮 Hasil Prediksi")
            
            with st.spinner("Memproses prediksi..."):
                predictions = dt_model.predict(uploaded_df_copy)
                probabilities = dt_model.predict_proba(uploaded_df_copy)
                
                # Create results dataframe
                results_df = uploaded_df.copy()
                results_df['Prediksi'] = predictions
                
                # Add probability columns
                for idx, class_name in enumerate(dt_model.classes_):
                    results_df[f'Probabilitas_{class_name}'] = probabilities[:, idx].round(4)
                
                # Add confidence
                results_df['Confidence'] = probabilities.max(axis=1).round(4)
            
            # Display results in table
            st.dataframe(results_df, use_container_width=True)
            
            st.divider()
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_predictions = len(results_df)
                st.metric("📊 Total Prediksi", total_predictions)
            
            with col2:
                rendah_count = (results_df['Prediksi'] == 'rendah').sum()
                st.metric("📉 Kategori Rendah", rendah_count)
            
            with col3:
                tinggi_count = (results_df['Prediksi'] == 'tinggi').sum()
                st.metric("📈 Kategori Tinggi", tinggi_count)
            
            st.divider()
            
            # Visualization
            st.subheader("📊 Distribusi Hasil Prediksi")
            
            pred_counts = results_df['Prediksi'].value_counts()
            colors = {'rendah': '#FF6B6B', 'tinggi': '#45B7D1'}
            color_list = [colors.get(cat, '#999999') for cat in pred_counts.index]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=pred_counts.index,
                    y=pred_counts.values,
                    marker=dict(color=color_list),
                    text=pred_counts.values,
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="Distribusi Kategori Prediksi",
                xaxis_title="Kategori Produktivitas",
                yaxis_title="Jumlah",
                template="plotly_white",
                hovermode='x unified'
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Download results
            st.subheader("💾 Download Hasil")
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=csv,
                file_name=f"batch_prediction_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Also provide Excel option if openpyxl is available
            try:
                import openpyxl
                from io import BytesIO
                
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='Predictions')
                    
                    # Format the Excel file
                    workbook = writer.book
                    worksheet = writer.sheets['Predictions']
                    
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Hasil Prediksi (Excel)",
                    data=excel_buffer,
                    file_name=f"batch_prediction_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.info("💡 Untuk download Excel, install openpyxl: `pip install openpyxl`")
        
        except Exception as e:
            st.error(f"❌ Error memproses file: {str(e)}")
            st.info("💡 Pastikan file CSV memiliki format yang benar dan semua kolom required ada")
    
    else:
        st.info("📌 Upload file CSV untuk memulai batch prediction")
        st.markdown("""
        ### 📋 Persyaratan File CSV:
        - Harus memiliki semua kolom feature yang diperlukan (21 variabel)
        - Format boolean columns: `True` atau `False`
        - Format numeric columns: angka (bisa integer atau float)
        - Kolom target (Yield_Category) bersifat opsional
        
        ### 📊 Fitur Batch Prediction:
        - ✅ Prediksi multiple data sekaligus
        - ✅ Probabilitas untuk setiap kelas
        - ✅ Confidence score
        - ✅ Visualisasi distribusi hasil
        - ✅ Download hasil (CSV/Excel)
        """)

def page_model_evaluation():
    """Model Evaluation Page"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {navy_blue} 0%, {light_navy} 100%); padding: 30px; border-radius: 10px; color: white; margin-bottom: 30px;">
        <h2 style="color: {accent_gold}; margin: 0;">📈 Model Evaluation</h2>
        <p style="margin: 10px 0 0 0;">Evaluasi performa model Decision Tree dan metrik terkait</p>
    </div>
    """, unsafe_allow_html=True)
    
    metrics, y_actual = get_model_performance()
    
    st.subheader("📊 Performa Model Decision Tree")
    
    # Metrics Comparison
    metrics_data = metrics['Decision Tree']
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics_data['accuracy']:.4f}", help="Tingkat akurasi model")
    with col2:
        st.metric("Precision", f"{metrics_data['precision']:.4f}", help="Presisi prediksi")
    with col3:
        st.metric("Recall", f"{metrics_data['recall']:.4f}", help="Recall/Sensitivity")
    with col4:
        st.metric("F1-Score", f"{metrics_data['f1']:.4f}", help="F1 Score")
    
    st.divider()
    
    # Detailed Metrics Table
    st.subheader("📋 Tabel Metrik Terperinci")
    dt_metrics_df = pd.DataFrame({
        'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Nilai': [f"{metrics['Decision Tree'][key]:.4f}" 
                 for key in ['accuracy', 'precision', 'recall', 'f1']]
    })
    st.dataframe(dt_metrics_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    
    cm_dt = confusion_matrix(y_actual, metrics['Decision Tree']['predictions'],
                            labels=['rendah', 'tinggi'])
    fig_cm_dt = go.Figure(data=go.Heatmap(
        z=cm_dt,
        x=['Rendah', 'Tinggi'],
        y=['Rendah', 'Tinggi'],
        colorscale='Greens',
        text=cm_dt,
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertemplate='Actual: %{y}, Predicted: %{x}, Count: %{z}<extra></extra>'
    ))
    fig_cm_dt.update_layout(
        title="Confusion Matrix - Decision Tree",
        xaxis_title="Prediksi",
        yaxis_title="Aktual",
        height=500
    )
    apply_plotly_theme(fig_cm_dt)
    st.plotly_chart(fig_cm_dt, use_container_width=True)
    
    st.divider()
    
    # Classification Report
    st.subheader("📑 Classification Report")
    dt_report = classification_report(
        y_actual, metrics['Decision Tree']['predictions'],
        output_dict=True, zero_division=0
    )
    dt_report_df = pd.DataFrame(dt_report).transpose()
    st.dataframe(dt_report_df.round(4), use_container_width=True)

# ==================== MAIN APP ====================
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: white;">🌾 Pertanian</h2>
            <p style="color: #FFD700; font-weight: bold;">Dashboard Prediksi Produktivitas</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        page = st.radio(
            "📍 Navigasi",
            [
                "🏠 Home",
                "📊 Dataset Overview",
                "🔍 EDA",
                "🤖 Single Prediction",
                "📤 Batch Prediction",
                "📈 Model Evaluation",
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("""
        #### ℹ️ Informasi
        - **Dataset**: Cleaned Agriculture
        - **Model**: Decision Tree
        - **Features**: 21 variabel
        - **Target**: Yield Category (2 kelas)
        """)
    
    # Route to appropriate page
    if page == "🏠 Home":
        page_home()
    elif page == "📊 Dataset Overview":
        page_dataset_overview()
    elif page == "🔍 EDA":
        page_eda()
    elif page == "🤖 Single Prediction":
        page_ml_prediction()
    elif page == "📤 Batch Prediction":
        page_batch_prediction()
    elif page == "📈 Model Evaluation":
        page_model_evaluation()

if __name__ == "__main__":
    main()
