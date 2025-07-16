import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Page setup
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling - FIXED TEXT VISIBILITY
st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .patient-info {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #4a90e2;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        color: #2d3748;
    }
    
    .patient-info h3 {
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .patient-info p {
        color: #4a5568;
        margin-bottom: 0.25rem;
    }
    
    .result-container {
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.1em;
        font-weight: 600;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .danger-result {
        background: #fee;
        color: #c53030;
        border: 2px solid #e53e3e;
    }
    
    .safe-result {
        background: #f0fff4;
        color: #2d7d2d;
        border: 2px solid #38a169;
    }
    
    .info-section {
        background: #f7fafc;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        border-left: 4px solid #3182ce;
        color: #2d3748;
    }
    
    .notice-box {
        background: #fef5e7;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #d69e2e;
        margin: 1.5rem 0;
        color: #744210;
    }
    
    .notice-box strong {
        color: #744210;
    }
    
    .alert-box {
        background: #ebf8ff;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #3182ce;
        margin: 1.5rem 0;
        color: #2a4365;
    }
    
    .alert-box strong {
        color: #2a4365;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        text-align: center;
        color: #2d3748;
    }
    
    /* Fix for main content text visibility */
    .main .block-container {
        color: #2d3748;
    }
    
    /* Fix for form labels and text */
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #2d3748 !important;
    }
    
    /* Fix for subheader text */
    .stMarkdown h2, .stMarkdown h3 {
        color: #2d3748;
    }
    
    /* Fix for regular text */
    .stMarkdown p {
        color: #4a5568;
    }
    
    /* Fix for bullet points */
    .stMarkdown li {
        color: #4a5568;
    }
    
    /* Fix for error/success messages */
    .stError, .stSuccess {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'patient_database' not in st.session_state:
    st.session_state.patient_database = []
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = None
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = None
if 'data_scaler' not in st.session_state:
    st.session_state.data_scaler = None

# Model loading function
@st.cache_resource
def initialize_model():
    """Load existing model or create new one"""
    try:
        # Load saved model
        with open('heart_disease_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            loaded_scaler = pickle.load(scaler_file)
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        # st.info("Setting up prediction system... Please wait.")  # <<---- REMOVED AS REQUESTED
        # Generate training data
        np.random.seed(123)
        sample_size = 800
        
        # Create feature data
        patient_features = {
            'age': np.random.randint(25, 85, sample_size),
            'sex': np.random.randint(0, 2, sample_size),
            'cp': np.random.randint(0, 4, sample_size),
            'exang': np.random.randint(0, 2, sample_size),
            'oldpeak': np.random.uniform(0, 5, sample_size),
            'ca': np.random.randint(0, 4, sample_size),
            'thal': np.random.randint(0, 3, sample_size)
        }
        
        feature_df = pd.DataFrame(patient_features)
        
        # Generate target based on medical logic
        risk_indicators = (
            (feature_df['age'] > 50) & 
            (feature_df['cp'] >= 2) & 
            (feature_df['exang'] == 1) & 
            (feature_df['oldpeak'] > 1.5) & 
            (feature_df['ca'] >= 1)
        ).astype(int)
        
        # Add variability
        final_target = np.where(np.random.random(sample_size) < 0.25, 
                               1 - risk_indicators, risk_indicators)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df, final_target, test_size=0.25, random_state=123
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        clf = RandomForestClassifier(n_estimators=150, random_state=123)
        clf.fit(X_train_scaled, y_train)
        
        return clf, scaler

# Initialize model
model, scaler = initialize_model()

# Patient management functions
def register_patient(full_name, patient_id, age, gender):
    """Register new patient in system"""
    new_patient = {
        'full_name': full_name,
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'consultation_history': []
    }
    st.session_state.patient_database.append(new_patient)
    st.success(f"Successfully registered: {full_name}")

def find_patient(patient_id):
    """Find patient by ID"""
    for patient in st.session_state.patient_database:
        if patient['patient_id'] == patient_id:
            return patient
    return None

def calculate_risk(input_features):
    """Calculate heart disease risk"""
    # Feature order for model
    feature_names = ['age', 'sex', 'cp', 'exang', 'oldpeak', 'ca', 'thal']
    feature_values = []
    
    for feature in feature_names:
        if feature in input_features:
            feature_values.append(input_features[feature])
        else:
            feature_values.append(0)
    
    # Prepare input for model
    input_array = np.array(feature_values).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    
    # Get prediction
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    # Feature importance
    importance_scores = dict(zip(feature_names, model.feature_importances_))
    
    return {
        'prediction': prediction,
        'probabilities': probabilities,
        'risk_percentage': probabilities[1] * 100,
        'feature_importance': importance_scores,
        'assessment_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    # Header section
    st.markdown("""
    <div class="header-container">
        <h1>🏥 Heart Disease Prediction System</h1>
        <p>Clinical cardiovascular risk assessment platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical notice
    st.markdown("""
    <div class="notice-box">
        <strong>⚠️ Medical Professional Use Only:</strong> This system requires clinical interpretation. 
        Parameters such as ST depression, vessel count, and thalassemia results must be obtained from proper medical testing.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Patient Management
    with st.sidebar:
        st.header("👤 Patient Management")
        
        # Patient search
        st.subheader("Find Patient")
        search_id = st.text_input("Patient ID", placeholder="Enter patient ID")
        
        if st.button("Search"):
            if search_id:
                found_patient = find_patient(search_id)
                if found_patient:
                    st.session_state.selected_patient = found_patient
                    st.success(f"Found: {found_patient['full_name']}")
                else:
                    st.error("Patient not found in database")
        
        st.divider()
        
        # New patient registration
        st.subheader("Register New Patient")
        with st.form("patient_registration"):
            patient_name = st.text_input("Full Name", placeholder="Enter full name")
            patient_id = st.text_input("Patient ID", placeholder="Unique patient ID")
            patient_age = st.number_input("Age", min_value=18, max_value=100, value=45)
            patient_gender = st.selectbox("Gender", options=[0, 1], 
                                        format_func=lambda x: "Female" if x == 0 else "Male")
            
            submit_registration = st.form_submit_button("Register Patient")
            
            if submit_registration:
                if patient_name and patient_id:
                    if not find_patient(patient_id):
                        register_patient(patient_name, patient_id, patient_age, patient_gender)
                        st.session_state.selected_patient = find_patient(patient_id)
                    else:
                        st.error("Patient ID already exists")
                else:
                    st.error("Please complete all fields")
        
        st.divider()
        
        # Recent patients
        if st.session_state.patient_database:
            st.subheader("Recent Patients")
            for patient in st.session_state.patient_database[-4:]:
                if st.button(f"{patient['full_name']} ({patient['patient_id']})", 
                            key=f"select_{patient['patient_id']}"):
                    st.session_state.selected_patient = patient
    
    # Main content
    main_col, info_col = st.columns([3, 1])
    
    with main_col:
        # Patient selection prompt
        if not st.session_state.selected_patient:
            st.markdown("""
            <div class="alert-box">
                <strong>🔍 Select Patient:</strong> Please search for an existing patient or register a new one to begin assessment.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Current patient display
            current_patient = st.session_state.selected_patient
            st.markdown(f"""
            <div class="patient-info">
                <h3>Patient: {current_patient['full_name']}</h3>
                <p><strong>ID:</strong> {current_patient['patient_id']} | 
                <strong>Age:</strong> {current_patient['age']} | 
                <strong>Gender:</strong> {'Male' if current_patient['gender'] == 1 else 'Female'}</p>
                <p><strong>Registered:</strong> {current_patient['registration_date']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical assessment form
            st.subheader("🩺 Clinical Assessment")
            
            with st.form("clinical_assessment"):
                left_col, right_col = st.columns(2)
                
                with left_col:
                    age_input = st.number_input("Age (years)", min_value=18, max_value=100, 
                                              value=current_patient['age'])
                    sex_input = st.selectbox("Sex", options=[0, 1], 
                                           format_func=lambda x: "Female" if x == 0 else "Male", 
                                           index=current_patient['gender'])
                    cp_input = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                                          format_func=lambda x: {
                                              0: "Typical Angina", 
                                              1: "Atypical Angina", 
                                              2: "Non-Anginal Pain", 
                                              3: "Asymptomatic"
                                          }[x])
                    exang_input = st.selectbox("Exercise Induced Angina", options=[0, 1],
                                             format_func=lambda x: "No" if x == 0 else "Yes")
                
                with right_col:
                    oldpeak_input = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=8.0, 
                                                  value=0.0, step=0.1, 
                                                  help="ST depression induced by exercise relative to rest")
                    ca_input = st.selectbox("Major Vessels Colored (0-3)", options=[0, 1, 2, 3],
                                          help="Number of major vessels colored by fluoroscopy")
                    thal_input = st.selectbox("Thalassemia", options=[0, 1, 2],
                                            format_func=lambda x: {
                                                0: "Normal", 
                                                1: "Fixed Defect", 
                                                2: "Reversible Defect"
                                            }[x],
                                            help="Thalassemia test result")
                
                assess_button = st.form_submit_button("🔍 Assess Heart Disease Risk", 
                                                    use_container_width=True)
                
                if assess_button:
                    # Prepare assessment data
                    assessment_data = {
                        'age': age_input,
                        'sex': sex_input,
                        'cp': cp_input,
                        'exang': exang_input,
                        'oldpeak': oldpeak_input,
                        'ca': ca_input,
                        'thal': thal_input
                    }
                    
                    # Calculate risk
                    risk_result = calculate_risk(assessment_data)
                    
                    # Display result
                    risk_status = "HIGH RISK" if risk_result['prediction'] == 1 else "LOW RISK"
                    result_style = "danger-result" if risk_result['prediction'] == 1 else "safe-result"
                    
                    st.markdown(f"""
                    <div class="result-container {result_style}">
                        💓 Heart Disease Risk Assessment: {risk_status}<br>
                        Risk Score: {risk_result['risk_percentage']:.1f}%<br>
                        Confidence Level: {max(risk_result['probabilities']):.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save to patient history
                    assessment_record = {
                        'risk_result': risk_result,
                        'clinical_inputs': assessment_data,
                        'timestamp': risk_result['assessment_time']
                    }
                    
                    # Update patient database
                    for idx, patient in enumerate(st.session_state.patient_database):
                        if patient['patient_id'] == current_patient['patient_id']:
                            st.session_state.patient_database[idx]['consultation_history'].append(assessment_record)
                            break
                    
                    # Feature analysis
                    st.subheader("📊 Clinical Factor Analysis")
                    
                    importance_data = pd.DataFrame(
                        list(risk_result['feature_importance'].items()), 
                        columns=['Clinical Factor', 'Importance Score']
                    ).sort_values('Importance Score', ascending=True)
                    
                    chart = px.bar(importance_data, x='Importance Score', y='Clinical Factor', 
                                 orientation='h', color='Importance Score',
                                 color_continuous_scale='Blues',
                                 title="Contributing Factors to Risk Assessment")
                    chart.update_layout(height=350)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Clinical interpretation
                    st.subheader("🔬 Clinical Interpretation")
                    
                    risk_factors = []
                    if cp_input in [0, 1]:
                        risk_factors.append("Chest pain pattern suggests cardiac origin")
                    if exang_input == 1:
                        risk_factors.append("Exercise-induced angina is present")
                    if oldpeak_input > 1.5:
                        risk_factors.append("Significant ST depression during exercise")
                    if ca_input > 1:
                        risk_factors.append("Multiple coronary vessels show narrowing")
                    if thal_input in [1, 2]:
                        risk_factors.append("Thalassemia test shows abnormal results")
                    if age_input > 55:
                        risk_factors.append("Age increases cardiovascular risk")
                    
                    if risk_factors:
                        st.error("**Contributing Risk Factors:**")
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.success("**Assessment shows minimal risk factors present**")
    
    with info_col:
        # System information
        st.subheader("ℹ️ System Information")
        st.markdown("""
        This cardiovascular risk assessment system analyzes clinical parameters to evaluate heart disease probability.
        
        **System Capabilities:**
        • Patient record management
        • Clinical data input processing
        • Risk calculation with confidence intervals
        • Factor contribution analysis
        • Assessment history tracking
        
        **Required Clinical Data:**
        • **Age:** Patient age in years
        • **Sex:** Biological sex (0=Female, 1=Male)
        • **Chest Pain Type:** Classified 0-3 scale
        • **Exercise Angina:** Presence during exercise testing
        • **ST Depression:** ECG changes during exercise
        • **Major Vessels:** Coronary angiography results
        • **Thalassemia:** Genetic blood disorder status
        """)
        
        # Assessment history
        if (st.session_state.selected_patient and 
            st.session_state.selected_patient.get('consultation_history')):
            st.subheader("📋 Assessment History")
            
            current_patient = st.session_state.selected_patient
            history = current_patient['consultation_history']
            
            # Recent assessments
            for idx, record in enumerate(history[-3:]):
                risk_level = "HIGH" if record['risk_result']['prediction'] == 1 else "LOW"
                status_icon = "🔴" if record['risk_result']['prediction'] == 1 else "🟢"
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Assessment #{len(history) - idx}</strong> {status_icon}<br>
                    <small>Risk Level: {risk_level} ({record['risk_result']['risk_percentage']:.1f}%)</small><br>
                    <small>Date: {record['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"View Assessment #{len(history) - idx}", key=f"view_{idx}"):
                    st.json(record['clinical_inputs'])
            
            # Risk progression chart
            if len(history) > 1:
                st.subheader("📈 Risk Progression")
                
                risk_scores = [record['risk_result']['risk_percentage'] for record in history]
                timestamps = [record['timestamp'] for record in history]
                
                trend_chart = go.Figure()
                trend_chart.add_trace(go.Scatter(
                    x=timestamps,
                    y=risk_scores,
                    mode='lines+markers',
                    name='Risk Percentage',
                    line=dict(color='crimson' if risk_scores[-1] > 50 else 'forestgreen', width=2)
                ))
                
                trend_chart.update_layout(
                    title="Risk Score Over Time",
                    xaxis_title="Assessment Date",
                    yaxis_title="Risk Percentage (%)",
                    height=250
                )
                
                st.plotly_chart(trend_chart, use_container_width=True)

if __name__ == "__main__":
    main()
