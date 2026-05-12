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
        
        [data-testid="stSidebar"] {{
            background-color: {navy_blue} !important;
        }}
        
        [data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] button {{
            background-color: {light_navy};
            color: {text_color};
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {accent_gold} !important;
            color: {navy_blue} !important;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {navy_blue} 0%, {light_navy} 100%);
            padding: 20px;
            border-radius: 10px;
            color: {text_color};
            border-left: 5px solid {accent_gold};
        }}
        
        h1, h2, h3 {{
            color: {navy_blue};
        }}
        
        .stButton > button {{
            background-color: {accent_gold};
            color: {navy_blue};
            font-weight: bold;
        }}
        
        .stButton > button:hover {{
            background-color: #FFC700;
        }}
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA & MODELS ====================
@st.cache_resource
def load_data_and_models():
    df = pd.read_csv('cleaned_agriculture.csv')
    nb_model = joblib.load('naive_bayes_model.pkl')
    dt_model = joblib.load('dt_smote_model.pkl')
    encoder = joblib.load('encoder.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return df, nb_model, dt_model, encoder, feature_columns

df, nb_model, dt_model, encoder, feature_columns = load_data_and_models()

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
    
    feature_array = np.array([[features_dict.get(col, 0) for col in feature_columns]])
    return feature_array

def get_model_performance():
    """Calculate model performance metrics on test set (80-20 split)"""
    X = df.drop('Yield_Category', axis=1)
    y = df['Yield_Category']
    
    # Split data 80-20 (train-test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Predictions on test set only
    nb_pred = nb_model.predict(X_test)
    dt_pred = dt_model.predict(X_test)
    
    metrics = {
        'Naive Bayes': {
            'accuracy': accuracy_score(y_test, nb_pred),
            'precision': precision_score(y_test, nb_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, nb_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, nb_pred, average='weighted', zero_division=0),
            'predictions': nb_pred
        },
        'Decision Tree SMOTE': {
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
        best_model = "Decision Tree SMOTE"
        metrics, _ = get_model_performance()
        best_acc = metrics[best_model]['accuracy']
        st.metric("🏆 Best Model Accuracy", f"{best_acc:.2%}", help="Model dengan performa terbaik")
    
    st.divider()
    
    # Key Features
    st.subheader("📋 Fitur Utama Dashboard")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔍 Analisis Data
        - **Dataset Overview**: Karakteristik data dan distribusi kelas
        - **EDA**: Visualisasi hubungan antar variabel
        - **Model Evaluation**: Perbandingan performa model
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 Machine Learning
        - **2 Model Klasifikasi**: Naive Bayes vs Decision Tree SMOTE
        - **Real-time Prediction**: Prediksi tanpa upload dataset
        - **Risk Assessment**: Analisis tingkat risiko produktivitas
        """)
    
    st.divider()
    
    # Model Information
    st.subheader("🧠 Model Machine Learning")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Naive Bayes
        - Model probabilistik sederhana
        - Asumsi fitur independen
        - Kecepatan prediksi tinggi
        """)
    
    with col2:
        st.markdown("""
        ### Decision Tree + SMOTE
        - Menangani data tidak seimbang
        - Interpretasi yang lebih mudah
        - Akurasi lebih tinggi
        """)
    
    st.divider()
    st.info("💡 **Tips**: Gunakan ML Prediction untuk melakukan simulasi prediksi produktivitas pertanian secara real-time!")

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
    st.plotly_chart(fig_adapt, use_container_width=True)

def page_eda():
    """EDA Page"""
    st.title("🔍 Exploratory Data Analysis")
    
    # Temperature Distribution
    st.subheader("🌡️ Distribusi Suhu Rata-rata")
    fig_temp = px.histogram(df, x='Average_Temperature_C', nbins=30,
                           title="Distribusi Suhu Rata-rata",
                           labels={'Average_Temperature_C': 'Suhu (°C)'})
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Precipitation Distribution
    st.subheader("🌧️ Distribusi Curah Hujan")
    fig_precip = px.histogram(df, x='Total_Precipitation_mm', nbins=30,
                             title="Distribusi Curah Hujan",
                             labels={'Total_Precipitation_mm': 'Curah Hujan (mm)'})
    st.plotly_chart(fig_precip, use_container_width=True)
    
    # Soil Health Distribution
    st.subheader("🌱 Distribusi Kesehatan Tanah")
    fig_soil = px.histogram(df, x='Soil_Health_Index', nbins=30,
                           title="Distribusi Indeks Kesehatan Tanah",
                           labels={'Soil_Health_Index': 'Soil Health Index'})
    st.plotly_chart(fig_soil, use_container_width=True)
    
    # Extreme Weather Events
    st.subheader("⛈️ Distribusi Peristiwa Cuaca Ekstrem")
    fig_extreme = px.histogram(df, x='Extreme_Weather_Events', nbins=30,
                              title="Distribusi Peristiwa Cuaca Ekstrem",
                              labels={'Extreme_Weather_Events': 'Kejadian Ekstrem'})
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
        st.plotly_chart(fig_temp_yield, use_container_width=True)
    
    with col2:
        fig_soil_yield = px.box(df, x='Yield_Category', y='Soil_Health_Index',
                               title="Kesehatan Tanah per Kategori Produktivitas",
                               color='Yield_Category',
                               color_discrete_map={'rendah': '#FF6B6B', 'tinggi': '#45B7D1'})
        st.plotly_chart(fig_soil_yield, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig_irrig_yield = px.box(df, x='Yield_Category', y='Irrigation_Access_%',
                                title="Akses Irigasi per Kategori Produktivitas",
                                color='Yield_Category',
                                color_discrete_map={'rendah': '#FF6B6B', 'tinggi': '#45B7D1'})
        st.plotly_chart(fig_irrig_yield, use_container_width=True)
    
    with col4:
        fig_co2_yield = px.box(df, x='Yield_Category', y='CO2_Emissions_MT',
                              title="Emisi CO₂ per Kategori Produktivitas",
                              color='Yield_Category',
                              color_discrete_map={'rendah': '#FF6B6B', 'tinggi': '#45B7D1'})
        st.plotly_chart(fig_co2_yield, use_container_width=True)

def page_ml_prediction():
    """ML Prediction Page"""
    st.title("🤖 Machine Learning Prediction")
    st.markdown("Simulasikan prediksi produktivitas pertanian dengan memasukkan parameter lingkungan")
    
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
        nb_pred = nb_model.predict(features)[0]
        dt_pred = dt_model.predict(features)[0]
        
        nb_proba = nb_model.predict_proba(features)[0]
        dt_proba = dt_model.predict_proba(features)[0]
        
        # Display Results
        st.subheader("📊 Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #001f3f 0%, #003d5c 100%); 
                        padding: 20px; border-radius: 10px; text-align: center; color: white;
                        border-left: 5px solid #FFD700;">
                <h4>Naive Bayes</h4>
                <h2 style="color: #FFD700;">{nb_pred.upper()}</h2>
                <p>Probabilitas: {max(nb_proba):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #001f3f 0%, #003d5c 100%); 
                        padding: 20px; border-radius: 10px; text-align: center; color: white;
                        border-left: 5px solid #FFD700;">
                <h4>Decision Tree SMOTE</h4>
                <h2 style="color: #FFD700;">{dt_pred.upper()}</h2>
                <p>Probabilitas: {max(dt_proba):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Probability Charts
        col1, col2 = st.columns(2)
        
        # Get model classes dan filter probabilities sesuai encoder
        model_classes = nb_model.classes_
        encoder_indices = [list(model_classes).index(cat) for cat in encoder if cat in model_classes]
        filtered_categories = [encoder[i] for i, cat in enumerate(encoder) if cat in model_classes]
        
        with col1:
            filtered_proba_nb = nb_proba[encoder_indices]
            nb_proba_df = pd.DataFrame({
                'Kategori': filtered_categories,
                'Probabilitas': filtered_proba_nb
            })
            fig_nb = px.bar(nb_proba_df, x='Kategori', y='Probabilitas',
                           title="Probabilitas Naive Bayes",
                           color='Kategori',
                           color_discrete_map={'rendah': '#FF6B6B', 'tinggi': '#45B7D1'})
            st.plotly_chart(fig_nb, use_container_width=True)
        
        with col2:
            filtered_proba_dt = dt_proba[encoder_indices]
            dt_proba_df = pd.DataFrame({
                'Kategori': filtered_categories,
                'Probabilitas': filtered_proba_dt
            })
            fig_dt = px.bar(dt_proba_df, x='Kategori', y='Probabilitas',
                           title="Probabilitas Decision Tree SMOTE",
                           color='Kategori',
                           color_discrete_map={'rendah': '#FF6B6B', 'tinggi': '#45B7D1'})
            st.plotly_chart(fig_dt, use_container_width=True)
        
        st.divider()
        
        # Risk Assessment
        st.subheader("⚠️ Analisis Risiko")
        risk_level = "RENDAH" if (nb_pred == "tinggi" and dt_pred == "tinggi") else "TINGGI"
        
        risk_color = "#45B7D1" if risk_level == "RENDAH" else "#FF6B6B"
        
        st.markdown(f"""
        <div style="background: {risk_color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h3>Tingkat Risiko: {risk_level}</h3>
            <p>Berdasarkan konsensus kedua model machine learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("💡 Rekomendasi")
        if risk_level == "RENDAH":
            st.success("✅ Kondisi pertanian optimal. Lanjutkan praktik pengelolaan lahan yang sudah diterapkan.")
        else:
            st.warning("⚠️ Risiko produktivitas tinggi. Pertimbangkan:\n- Peningkatan akses irigasi\n- Perbaikan kesehatan tanah\n- Penerapan strategi adaptasi yang lebih baik")

def page_model_evaluation():
    """Model Evaluation Page"""
    st.title("📈 Model Evaluation")
    
    metrics, y_actual = get_model_performance()
    
    st.subheader("📊 Perbandingan Performa Model")
    
    # Metrics Comparison
    metrics_comparison = pd.DataFrame({
        'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Naive Bayes': [
            metrics['Naive Bayes']['accuracy'],
            metrics['Naive Bayes']['precision'],
            metrics['Naive Bayes']['recall'],
            metrics['Naive Bayes']['f1']
        ],
        'Decision Tree SMOTE': [
            metrics['Decision Tree SMOTE']['accuracy'],
            metrics['Decision Tree SMOTE']['precision'],
            metrics['Decision Tree SMOTE']['recall'],
            metrics['Decision Tree SMOTE']['f1']
        ]
    })
    
    fig_comparison = go.Figure(data=[
        go.Bar(name='Naive Bayes', x=metrics_comparison['Metrik'], 
               y=metrics_comparison['Naive Bayes'], marker_color='#45B7D1'),
        go.Bar(name='Decision Tree SMOTE', x=metrics_comparison['Metrik'],
               y=metrics_comparison['Decision Tree SMOTE'], marker_color='#FFD700')
    ])
    fig_comparison.update_layout(
        barmode='group',
        title="Perbandingan Metrik Evaluasi",
        xaxis_title="Metrik",
        yaxis_title="Nilai",
        template="plotly_white"
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed Metrics Table
    st.subheader("📋 Tabel Metrik Terperinci")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Naive Bayes")
        nb_metrics_df = pd.DataFrame({
            'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Nilai': [f"{metrics['Naive Bayes'][key]:.4f}" 
                     for key in ['accuracy', 'precision', 'recall', 'f1']]
        })
        st.dataframe(nb_metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("### Decision Tree SMOTE")
        dt_metrics_df = pd.DataFrame({
            'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Nilai': [f"{metrics['Decision Tree SMOTE'][key]:.4f}" 
                     for key in ['accuracy', 'precision', 'recall', 'f1']]
        })
        st.dataframe(dt_metrics_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Confusion Matrices
    st.subheader("🎯 Confusion Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cm_nb = confusion_matrix(y_actual, metrics['Naive Bayes']['predictions'], 
                                labels=['rendah', 'tinggi'])
        fig_cm_nb = go.Figure(data=go.Heatmap(
            z=cm_nb,
            x=['Rendah', 'Tinggi'],
            y=['Rendah', 'Tinggi'],
            colorscale='Blues',
            text=cm_nb,
            texttemplate='%{text}',
            textfont={"size": 14},
            hovertemplate='Actual: %{y}, Predicted: %{x}, Count: %{z}<extra></extra>'
        ))
        fig_cm_nb.update_layout(
            title="Confusion Matrix - Naive Bayes",
            xaxis_title="Prediksi",
            yaxis_title="Aktual",
            height=400
        )
        st.plotly_chart(fig_cm_nb, use_container_width=True)
    
    with col2:
        cm_dt = confusion_matrix(y_actual, metrics['Decision Tree SMOTE']['predictions'],
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
            title="Confusion Matrix - Decision Tree SMOTE",
            xaxis_title="Prediksi",
            yaxis_title="Aktual",
            height=400
        )
        st.plotly_chart(fig_cm_dt, use_container_width=True)
    
    st.divider()
    
    # Classification Reports
    st.subheader("📑 Classification Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Naive Bayes")
        nb_report = classification_report(
            y_actual, metrics['Naive Bayes']['predictions'],
            output_dict=True, zero_division=0
        )
        nb_report_df = pd.DataFrame(nb_report).transpose()
        st.dataframe(nb_report_df.round(4), use_container_width=True)
    
    with col2:
        st.write("### Decision Tree SMOTE")
        dt_report = classification_report(
            y_actual, metrics['Decision Tree SMOTE']['predictions'],
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
            ["🏠 Home", "📊 Dataset Overview", "🔍 EDA", "🤖 ML Prediction", "📈 Model Evaluation"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("""
        #### ℹ️ Informasi
        - **Dataset**: Cleaned Agriculture
        - **Models**: Naive Bayes & Decision Tree SMOTE
        - **Features**: 21 variabel
        - **Target**: Yield Category (3 kelas)
        """)
    
    # Route to appropriate page
    if page == "🏠 Home":
        page_home()
    elif page == "📊 Dataset Overview":
        page_dataset_overview()
    elif page == "🔍 EDA":
        page_eda()
    elif page == "🤖 ML Prediction":
        page_ml_prediction()
    elif page == "📈 Model Evaluation":
        page_model_evaluation()

if __name__ == "__main__":
    main()
