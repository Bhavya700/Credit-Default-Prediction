# Credit Default Prediction Project

## Overview
This project implements a comprehensive machine learning pipeline for predicting credit default risk using Lending Club data. The project is structured in four main phases, each focusing on different aspects of the machine learning workflow.

## Project Structure
```
Credit Default Prediction/
├── EDA-Preprocessing.ipynb                    # Phase 1: Exploratory Data Analysis & Preprocessing
├── Handling-Imbalanced-Data.ipynb             # Phase 2: Data Balancing & Class Weight Calculation
├── Model.ipynb                                # Phase 3: Model Training & Evaluation
├── Model_Evaluation-Results_Interpretation.ipynb  # Phase 4: Model Evaluation & Results Interpretation
├── processed_lending_club_data.parquet        # Clean, preprocessed dataset
├── trained_models.joblib                      # Trained models (not in git due to size)
├── README.md                                  # Project documentation
└── requirements.txt                           # Python dependencies
```

## Dataset
- **Source**: Lending Club loan data (2007-2018)
- **Size**: ~2.2M loan records
- **Features**: 96 engineered features
- **Target**: Binary classification (0: Fully Paid, 1: Charged Off)
- **Class Distribution**: Highly imbalanced (80% Fully Paid, 20% Charged Off)

## Key Features
- **Stratified Sampling**: Maintains class distribution across train/test splits
- **SMOTE**: Synthetic Minority Over-sampling Technique for balanced training
- **Class Weights**: Alternative approach to handle class imbalance
- **Feature Engineering**: Comprehensive feature creation and selection
- **Performance Metrics**: AUC-ROC, Precision, Recall, F1-Score

## Results
- Comprehensive feature engineering pipeline
- Balanced dataset preparation
- Multiple model evaluation approaches
- Production-ready preprocessing workflow

## Model Performance Results

### Best Performing Model
**LightGBM (Weighted)** achieved the highest performance across all metrics:
- **Recall**: 0.9972 (99.72%)
- **Precision**: 0.9986 (99.86%)
- **F1-Score**: 0.9979 (99.79%)
- **AUC**: 0.9996 (99.96%)

### Complete Model Comparison
| Model | Precision | Recall | F1-Score | AUC |
|-------|-----------|---------|-----------|-----|
| LightGBM (Weighted) | 0.9986 | **0.9972** | **0.9979** | 0.9996 |
| LightGBM (SMOTE) | 0.9998 | 0.9960 | 0.9979 | **0.9999** |
| Logistic Regression (SMOTE) | 0.9986 | 0.9893 | 0.9939 | 0.9995 |
| Random Forest (Weighted) | 1.0000 | 0.9888 | 0.9943 | 0.9999 |
| Random Forest (SMOTE) | 0.9999 | 0.9886 | 0.9943 | 0.9999 |
| Logistic Regression (Weighted) | 0.9979 | 0.9837 | 0.9907 | 0.9990 |

### Key Findings
1. **Excellent Performance**: All models achieved exceptional performance with AUC > 99.9%
2. **High Recall**: The best model correctly identifies 99.72% of actual defaulters
3. **Balanced Performance**: Both class weighting and SMOTE techniques performed well
4. **Feature Importance**: Top predictive features include:
   - `total_rec_prncp`: Total principal received
   - `funded_amnt_inv`: Amount funded by investors
   - `funded_amnt`: Total amount funded
   - `last_pymnt_amnt`: Last payment amount
   - `last_fico_range_high`: Latest FICO score

### Business Implications
- **Risk Mitigation**: High recall ensures minimal missed defaults, protecting against credit losses
- **Operational Efficiency**: High precision reduces false positives, minimizing unnecessary loan rejections
- **Underwriting Insights**: Feature importance analysis guides risk assessment criteria
- **Model Deployment**: Ready for production use in real-time loan scoring

## Technologies Used
- Python 3.x
- Pandas, NumPy for data manipulation
- Scikit-learn for machine learning
- Imbalanced-learn for SMOTE
- Matplotlib, Seaborn for visualization
- Jupyter Notebooks for development

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in sequence:
   - Start with `EDA-Preprocessing.ipynb`
   - Continue with `Handling-Imbalanced-Data.ipynb`
   - Finish with `Model.ipynb`


## Author
Bhavya Patel

## License
This project is open source and available under the MIT License.
