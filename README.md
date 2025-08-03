# SMS Spam Classifier using NLP
This project demonstrates how to build a robust machine learning pipeline for classifying messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques and various machine learning models. It includes exploratory data analysis, preprocessing, feature engineering, model training, and evaluation.

# 📁 Dataset
● Source: spam.csv

● Columns Used:

  ○ v1: Label (spam or ham)

  ○ v2: Message text

● Processed to:

  ○ is_spam: Binary label (1 = spam, 0 = ham)

  ○ text: Cleaned message content

# 📊 Exploratory Data Analysis (EDA)
● Distribution of spam vs. ham messages

● Word frequency analysis for both classes

● Word count histograms

● Correlation heatmap between message length and spam label

# 🔄 Text Preprocessing
Text cleaning steps include:

● Lowercasing

● Expanding contractions

● Removing punctuation and special characters

● Removing stopwords (via spaCy)

● Tokenization and lemmatization

# 🧠 Feature Engineering
● Message length (sent_len(in words))

● TF-IDF Vectorization of text data

● Custom preprocessing pipeline using FunctionTransformer

# 🧪 Models Implemented
Multiple classifiers were trained and compared:

● Gaussian Naive Bayes

● Multinomial Naive Bayes

● Bernoulli Naive Bayes

● Logistic Regression

● K-Nearest Neighbors

● Decision Tree

● Random Forest

● Support Vector Machine (SVM)

# 📈 Model Evaluation
● F1 Score

● Classification Report

● Cross-Validation using cross_val_score

● Visualizations of model performance and distribution

# 📦 Tech Stack
● Python 

● Jupyter Notebook

● Pandas, Matplotlib, Seaborn

● Scikit-learn

● spaCy (for NLP)

● Optuna (for tuning)

# 📁 Project Structure
├── spam.csv            # Main dataset with SMS messages and labels
├── slang_words.csv     # Custom dictionary of slang word mappings
├── sms-spam-detection.ipynb              # Jupyter notebooks for exploration and modeling 
├── requirements.txt        # Dependencies for running the project
└── README.md               # Project overview and setup instructions


# 🚀 How to Run
1. Clone the repository:

```
git clone https://github.com/rajjaiswal159/SMS_Spam_Classification_Using_NLP.git
cd SMS_Spam_Classification_Using_NLP
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Download spaCy English model:

```
python -m spacy download en_core_web_sm
```

4. Launch the notebook:

```
jupyter notebook sms-spam-detection.ipynb
```
