# Credit Default Prediction Project

## Overview
This project implements a comprehensive machine learning pipeline for predicting credit default risk using Lending Club data. The project is structured in three main phases, each focusing on different aspects of the machine learning workflow.

## Project Structure
```
Credit Default Prediction/
├── EDA-Preprocessing.ipynb          # Phase 1: Exploratory Data Analysis & Preprocessing
├── Handling-Imbalanced-Data.ipynb   # Phase 2: Data Balancing & Class Weight Calculation
├── Model.ipynb                      # Phase 3: Model Training & Evaluation
├── processed_lending_club_data.parquet  # Clean, preprocessed dataset
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

## Dataset
- **Source**: Lending Club loan data (2007-2018)
- **Size**: ~2.2M loan records
- **Features**: 96 engineered features
- **Target**: Binary classification (0: Fully Paid, 1: Charged Off)
- **Class Distribution**: Highly imbalanced (80% Fully Paid, 20% Charged Off)

## Phases

### Phase 1: EDA & Preprocessing
- Comprehensive data exploration and visualization
- Feature engineering and selection
- Data cleaning and preprocessing
- Handling missing values and outliers
- Data type conversions and encoding

### Phase 2: Handling Imbalanced Data
- Data splitting with stratification
- SMOTE implementation for oversampling
- Class weight calculation for imbalanced learning
- Data export for model training

### Phase 3: Model Training & Evaluation
- Multiple algorithm implementation
- Hyperparameter tuning
- Performance evaluation metrics
- Model comparison and selection

## Key Features
- **Stratified Sampling**: Maintains class distribution across train/test splits
- **SMOTE**: Synthetic Minority Over-sampling Technique for balanced training
- **Class Weights**: Alternative approach to handle class imbalance
- **Feature Engineering**: Comprehensive feature creation and selection
- **Performance Metrics**: AUC-ROC, Precision, Recall, F1-Score

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

## Results
- Comprehensive feature engineering pipeline
- Balanced dataset preparation
- Multiple model evaluation approaches
- Production-ready preprocessing workflow

## Author
Bhavya Patel

## License
This project is open source and available under the MIT License.
