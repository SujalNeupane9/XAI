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
- **new model responses/**: Additional data for evaluating AI models
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

## Weighted Average Ensemble Method

The `oof-and-ensemble-of-best-performing-models.ipynb` notebook implements an advanced ensemble technique that combines predictions from multiple models trained on different datasets:

### How It Works:

1. **Multi-Dataset Training**: Best-performing models are trained on three different datasets:
   - **COLING Dataset**: Uses Random Forest (best model for COLING)
   - **PAN CLEF Dataset**: Uses XGBoost (best model for PAN CLEF)
   - Models are trained on engineered features (30 features including character count, word count, sentiment, grammar errors, etc.)

2. **Out-of-Fold (OOF) Predictions**: 
   - Uses 5-fold cross-validation during training
   - Captures prediction probabilities for both classes from each fold
   - Generates both validation predictions and test predictions for all models

3. **Probability Stacking**:
   - Combines probability predictions from models trained on different datasets
   - Creates feature matrices with probabilities from both COLING and PAN CLEF trained models
   - This creates a meta-learning approach where models learn from each other's outputs

4. **Ridge Regression Weighting**:
   - Applies Ridge regression to learn optimal weights for combining ensemble predictions
   - Trains on a subset of probability predictions
   - Automatically learns which model/dataset combination contributes most to final predictions
   - Uses the learned weights to generate final predictions by threshold (0.5)

### Results:
- Tests the ensemble on multiple data sources (Ghostbuster Wikipedia, Reuters, Essays)
- Evaluates performance using F1 score metrics
- Compares cross-validation performance across multiple datasets

This approach leverages the strengths of each model while using Ridge regression to find the optimal weighted combination, resulting in improved generalization performance.
