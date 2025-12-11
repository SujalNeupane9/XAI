# XAI - AI Text Detection Project

This project focuses on AI-generated text detection and classification using various machine learning models and advanced feature engineering techniques.

## Project Structure

### `/datasets`
Contains datasets used for training and evaluation:
- **COLING 2025/**: COLING competition dataset
  - `coling_train.csv` - Training dataset
  - `coling_test3000.csv` - Test dataset with 3000 samples
- **PAN CLEF/**: PAN CLEF challenge dataset
  - `test.csv` - Test dataset
- **new model responses/**: Additional data
  - `new_data.csv` - New model responses data

### `/models`
Pre-trained machine learning models organized by dataset:
- **model trained on optimal feature set of PAN CLEF/**
  - Logistic Regression model
  - Random Forest model
  - SVM model
  - XGB (XGBoost) model
- **models trained on optimal feature set of COLING train/**
  - Logistic Regression model
  - Random Forest model
  - SVM model
  - XGB (XGBoost) model

### `/notebook`
Jupyter notebooks for data analysis, feature engineering, and model evaluation:
- `ai-text-detection-pos-tag-perplexity-gram-errors.ipynb` - Analysis using POS tags, perplexity, and grammar errors
- `classification-using-embeddings.ipynb` - Classification using text embeddings
- `creating-additional-features-for-pan-clef.ipynb` - Feature engineering for PAN CLEF dataset
- `creating-new-features-with-train-split-coling.ipynb` - Feature engineering for COLING with train/split
- `creating-with-new-features-coling.ipynb` - COLING dataset feature creation
- `evaluation-with-new-features-pan-clef.ipynb` - Model evaluation on PAN CLEF with new features
- `making-features-in-pan-clef-dataset.ipynb` - Initial feature creation for PAN CLEF
- `pretraining-data-analysis.ipynb` - Pre-training data analysis
- **notebook for weighted average/**
  - `oof-and-ensemble-of-best-performing-models.ipynb` - Ensemble models and weighted averaging

## Key Features

- **Multiple Models**: Logistic Regression, Random Forest, SVM, and XGBoost classifiers
- **Advanced Feature Engineering**: POS tagging, perplexity analysis, grammar error detection, text embeddings
- **Multiple Datasets**: Training and evaluation on COLING 2025 and PAN CLEF datasets
- **Ensemble Methods**: Weighted averaging of best-performing models

## Usage

Run the notebooks in the `/notebook` directory to:
1. Analyze data and create features
2. Train models on optimal feature sets
3. Evaluate model performance
4. Create ensemble predictions

Pre-trained models are available in the `/models` directory for direct inference.
