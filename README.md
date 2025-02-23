# Task 2: SMS Spam Classification

This repository contains a Python-based spam classification system built using supervised learning. The goal is to classify SMS messages as "spam" or "ham" (not spam) using the SMS Spam Collection dataset from Kaggle. The system employs TF-IDF for feature extraction and Logistic Regression for classification, with performance evaluation metrics included.

## Dataset

- **Source**: [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description**: A collection of 5,572 SMS messages labeled as "spam" or "ham."
- **File**: `spam.csv` (included in this repository)
- **Columns**:
  - `v1`: Label ("ham" or "spam")
  - `v2`: Text of the SMS message

## Project Structure

```
Task2-SpamClassification/
├── spam_classifier.py # Main script for training, evaluation, and prediction
├── spam.csv # Dataset file
├── README.md # Project documentation (this file)
├── requirements.txt # Python dependencies
└── .gitignore # Git ignore file (optional)
```

## How It Works

1. **Data Preprocessing**:
   - Loads the `spam.csv` dataset using pandas.
   - Converts labels ("ham" → 0, "spam" → 1).
   - Uses `TfidfVectorizer` to transform text into numerical features, removing English stop words and limiting to 5,000 features.
2. **Model Training**:
   - Splits data into training (80%) and testing (20%) sets.
   - Trains a Logistic Regression model on the training data.
3. **Evaluation**:
   - Computes accuracy and provides a classification report (precision, recall, F1-score) for "ham" and "spam" classes.
4. **Prediction**:
   - Classifies new example messages provided in the script.

## Requirements

To run this project, install the dependencies listed in `requirements.txt`:

## Usage

Clone the repository:

```bash
git clone https://github.com/Macmilan24/Task2-SpamClassification.git
cd Task2-SpamClassification
```

Ensure `spam.csv` is present (included).

Run the script:

```bash
python spam_classifier.py
```

Output includes:

- Model accuracy
- Classification report
- Predictions for example messages
