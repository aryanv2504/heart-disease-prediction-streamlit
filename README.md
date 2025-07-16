# Heart Disease Prediction System 🫀

A machine learning-powered heart disease prediction system built with Streamlit, trained on the Cleveland Heart Disease dataset. The system compares three different ML algorithms (Logistic Regression, Random Forest, and XGBoost) and uses the best-performing model for accurate heart disease predictions.

## 🌐 **LIVE DEMO - TRY THE APP NOW!**
# **[🚀 Heart Disease Prediction App - LIVE](https://heart-disease-prediction-app-byaryan.streamlit.app/)**

![App Interface View 1](assets/SCREENSHOT%20-%20APP%201.jpg)    
                                                                 ![App Interface View 2](assets/SCREENSHOT-%20APP-2.jpg)

---

## 🚀 Features

### 🤖 Machine Learning Models
- **Logistic Regression**: Best performing model with highest accuracy
- **Random Forest**: Ensemble learning method for comparison
- **XGBoost**: Gradient boosting algorithm for robust predictions
- **Model Comparison**: Comprehensive evaluation of all three algorithms

- ![ROC Curve](assets/SCREENSHOT%20-%20ROC%20CURVE.jpg)
- 

### 🏥 Patient Management System
- **Patient Registration**: Register new patients with demographic information
- **Patient Search**: Quickly find existing patients by name or ID
- **Patient Records**: Maintain comprehensive patient histories
- **Recent Patients**: Easy access to recently assessed patients

### 🩺 Clinical Assessment Engine
The system analyzes key clinical parameters from the Cleveland Heart Disease dataset:
- **Age**: Patient age in years
- **Sex**: Biological sex (0=Female, 1=Male)
- **Chest Pain Type (CP)**: 
  - 0: Typical Angina
  - 1: Atypical Angina
  - 2: Non-anginal Pain
  - 3: Asymptomatic
- **Thalassemia**: Blood disorder status
- **Exercise Induced Angina**: Presence during exercise testing
- **ST Depression**: Exercise-induced ST depression levels
- **Major Vessels Count**: Number of major vessels (0-4)

### 📊 Advanced Analytics & Reporting
- **Risk Score Calculation**: Precise probability percentage with confidence intervals
- **Clinical Factor Analysis**: Visual representation of contributing risk factors
- **Feature Importance**: Shows which clinical factors contribute most to prediction
- **Risk Assessment History**: Track patient assessments over time
- **Model Performance Metrics**: Accuracy, precision, recall, and F1-score

## 🛠️ Technical Architecture

### Machine Learning Pipeline
```
Cleveland Heart Disease Dataset
          ↓
Data Preprocessing & Feature Engineering
          ↓
Model Training & Comparison:
├── Logistic Regression ⭐ (Best Model)
├── Random Forest
└── XGBoost
          ↓
Model Evaluation & Selection
          ↓
Model Serialization (Pickle)
          ↓
Streamlit Web Application
```

### Dataset Information
- **Source**: Cleveland Heart Disease Dataset
- **Features**: 13 clinical attributes
- **Target**: Binary classification (0=No Disease, 1=Disease)
- **Preprocessing**: Feature scaling, encoding, and validation

### Model Performance
- **Logistic Regression**: Best accuracy (selected for deployment)
- **Random Forest**: Good performance with feature importance insights
- **XGBoost**: Robust gradient boosting performance
- **Evaluation Metrics**: Cross-validation, confusion matrix, ROC curve

## 🔧 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/aryanv2504/heart-disease-prediction-streamlit.git
cd heart-disease-prediction-streamlit
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the application**
```
Open your browser and navigate to: http://localhost:8501
```

## 🌐 **OR TRY THE LIVE DEPLOYED VERSION:**
# **[https://heart-disease-prediction-app-byaryan.streamlit.app/](https://heart-disease-prediction-app-byaryan.streamlit.app/)**

### Required Dependencies
```python
streamlit
pandas
numpy
scikit-learn
xgboost
pickle
plotly
matplotlib
seaborn
```

## 💻 Usage Guide

### 1. Patient Registration
- Navigate to "Register New Patient" section
- Enter patient details (name, ID, age, gender)
- Click "Register Patient" to create new record

### 2. Clinical Assessment
Input the following clinical parameters:
- **Age**: Patient age in years
- **Sex**: Select Male/Female
- **Chest Pain Type**: Choose from 4 types
- **ST Depression**: Exercise-induced ST depression (0-6.2)
- **Major Vessels Count**: Number of vessels (0-4)
- **Thalassemia**: Blood disorder status
- **Exercise Induced Angina**: Yes/No

### 3. Heart Disease Prediction
- Click "Assess Heart Disease Risk"
- View results including:
  - **Risk Score**: Probability percentage (e.g., 43.3%)
  - **Confidence Level**: Model confidence (e.g., 0.6%)
  - **Risk Classification**: LOW/MODERATE/HIGH
  - **Contributing Factors**: Visual analysis of risk factors

## 📊 Model Development Process

### 1. Data Preparation
```python
# Cleveland Heart Disease Dataset
- Data cleaning and preprocessing
- Feature engineering and selection
- Train-test split (80-20)
- Feature scaling using StandardScaler
```

### 2. Model Training & Comparison
```python
# Three models trained and compared:
1. Logistic Regression (sklearn) ⭐ Best Performance
2. Random Forest Classifier (sklearn)
3. XGBoost Classifier (xgboost)
```

### 3. Model Evaluation & Visualization
The project includes comprehensive model evaluation with visual analysis:

#### Confusion Matrix Analysis
- **Logistic Regression Confusion Matrix**: Shows best classification performance
- **Random Forest Confusion Matrix**: Ensemble method evaluation
- **XGBoost Confusion Matrix**: Gradient boosting performance metrics
![Logistic Regression Confusion Matrix](assets/SCREENSHOT-LOGISTIC%20REGRESSION-CONFUSION%20MATRIX.jpg)     ![Random Forest Confusion Matrix](assets/SCREENSHOT-RANDOM%20FOREST-CONFUSION%20MATRIX.jpg)
![XGBoost Confusion Matrix](assets/SCREENSHOT-XG%20BOOST-CONFUSION%20MATRIX.jpg)
#### Performance Visualization
- **ROC Curve Analysis**: Receiver Operating Characteristic curve comparison
- **Feature Distribution**: Analysis of feature importance and distribution
- **Best Model Comparison**: Visual comparison of all three models

#### Key Metrics Evaluated
```python
# Performance metrics for all models:
- Accuracy Score
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix Analysis
- Cross-validation scores
```

### 4. Model Selection
- **Logistic Regression** achieved the best performance across all metrics
- Superior accuracy and precision compared to Random Forest and XGBoost
- Model saved using pickle for deployment
- Integrated into Streamlit application for real-time predictions

## 🎯 Cleveland Heart Disease Dataset

### Dataset Features
1. **Age**: Age in years
2. **Sex**: Gender (1=male, 0=female)
3. **CP**: Chest pain type (0-3)
4. **Trestbps**: Resting blood pressure
5. **Chol**: Serum cholesterol
6. **Fbs**: Fasting blood sugar
7. **Restecg**: Resting ECG results
8. **Thalach**: Maximum heart rate achieved
9. **Exang**: Exercise induced angina
10. **Oldpeak**: ST depression induced by exercise
11. **Slope**: Slope of peak exercise ST segment
12. **Ca**: Number of major vessels (0-4)
13. **Thal**: Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)

### Target Variable
- **0**: No heart disease
- **1**: Heart disease present

## 🔒 Medical Disclaimer

**⚠️ Important:** This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

## 📈 System Performance

### Model Comparison Results
- **Logistic Regression**: ⭐ Best overall performance
- **Random Forest**: Good interpretability with feature importance
- **XGBoost**: Robust performance with gradient boosting

### Risk Assessment Accuracy
- High precision in heart disease prediction
- Low false positive rates
- Comprehensive confidence interval reporting
- Feature importance analysis for clinical insights

## 🚀 Deployment

### Streamlit Cloud Deployment
The application is deployed on Streamlit Cloud for easy access:

## 🌐 **[LIVE APP - CLICK HERE TO TEST](https://heart-disease-prediction-app-byaryan.streamlit.app/)**

- **Automatic Updates**: Connected to GitHub for continuous deployment
- **Scalable**: Handles multiple concurrent users
- **Real-time Predictions**: Instant heart disease risk assessment

### Local Development
```bash
# Run locally for development
streamlit run app.py
```

## 📁 Repository Structure
```
heart-disease-prediction-streamlit/
├── README.md
├── requirements.txt
├── app.py                                    # Main Streamlit application
├── model/
│   ├── heart_disease_model.pkl              # Trained Logistic Regression model
│   └── model_training.ipynb                 # Jupyter notebook with model training
├── data/
│   └── cleveland_heart_disease.csv          # Dataset
├── screenshots/
│   ├── SCREENSHOT_APP_RESULTS.jpg           # App prediction results
│   ├── SCREENSHOT_-_APP_-3.jpg              # App interface view 3
│   ├── SCREENSHOT_-_APP_1.jpg               # App interface view 1
│   ├── SCREENSHOT_-_ROC_CURVE.jpg           # ROC curve analysis
│   ├── SCREENSHOT-_APP-2.jpg                # App interface view 2
│   ├── SCREENSHOT-BEST_MODEL.jpg            # Best model comparison
│   ├── SCREENSHOT-FEATURE_DISTRIBUTION.jpg  # Feature distribution analysis
│   ├── SCREENSHOT-LOGISTIC_REGRESSION-CONFUSION_MATRIX.jpg # LR confusion matrix
│   ├── SCREENSHOT-RANDOM_FOREST-CONFUSION_MATRIX.jpg       # RF confusion matrix
│   └── SCREENSHOT-XG_BOOST-CONFUSION_MATRIX.jpg            # XGBoost confusion matrix
└── utils/
    ├── data_processing.py                   # Data preprocessing functions
    └── model_utils.py                       # Model loading and prediction functions
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Developer

**Aryan Verma**
- GitHub: [@aryanv2504](https://github.com/aryanv2504)
- Portfolio: [Your Portfolio](https://your-portfolio-url.com)

## 🔄 Version History

- **v1.0.0**: Initial release with comprehensive heart disease prediction system
  - Logistic Regression model deployment
  - Patient management system
  - Clinical assessment interface
  - Risk factor analysis and visualization
  - Streamlit Cloud deployment

## 🆘 Support

For support, please:
- Open an issue on GitHub
- Check the documentation
- Contact the developer

---

**⚡ Quick Start:**
1. Clone repository: `git clone https://github.com/aryanv2504/heart-disease-prediction-streamlit.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run application: `streamlit run app.py`
4. Open browser: `http://localhost:8501`

**🎯 Key Achievement:** Logistic Regression achieved the best performance among all three models tested, providing accurate and reliable heart disease predictions based on clinical parameters.

---

*This machine learning system uses the Cleveland Heart Disease dataset to predict heart disease risk. The Logistic Regression model was selected after comprehensive comparison with Random Forest and XGBoost algorithms.*
