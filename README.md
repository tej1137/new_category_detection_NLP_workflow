News Category Classification - End-to-End NLP Pipeline

Project Overview
This project implements an end-to-End Machine Learning pipeline to classify news headlines into distinct categories (e.g., Politics, Wellness, Entertainment). Utilizing a dataset of over 200,000 news articles, the project explores various Natural Language Processing (NLP) techniques, vectorization methods, and classification algorithms to identify the most effective approach for text classification.

The workflow includes comprehensive data cleaning, exploratory data analysis (EDA), handling class imbalance via down-sampling, and rigorous model evaluation.

Key Features
Data Preprocessing: Implementation of text normalization, stop-word removal, stemming (Porter), and lemmatization (WordNet) to clean raw text.

EDA & Visualization: Insights into data distribution using bar plots, box plots, and word clouds to identify class imbalances and text characteristics.

Vectorization Strategies: Comparative analysis of TF-IDF, Bag of Words (BoW), and Hashing Vectorization.

Model Comparison: Benchmarking five different classifiers:

Random Forest

Multinomial Naive Bayes (MNB)

Logistic Regression

Support Vector Machines (SVM)

K-Nearest Neighbors (KNN)

Optimization: Achieved ~93.7% accuracy by optimizing feature selection (n-grams) and focusing on high-volume categories (Politics, Wellness, Entertainment).

Language: Python

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn (Sklearn)

NLP: NLTK (WordNetLemmatizer, PorterStemmer)

Visualization: Matplotlib, Seaborn, WordCloud

Methodology & Results
1. Initial Experimentation (Multi-Class)
Initial tests were conducted on a balanced subset of data across multiple categories using 500 samples per class.

Best Vectorizer: TF-IDF consistently outperformed BoW and Hashing.

Best Model: Logistic Regression achieved the highest accuracy of 69.38%, closely followed by SVM and Multinomial NB.

2. Optimized Approach (High-Accuracy)
To test the limits of the models, the scope was refined to the top 3 categories (Politics, Wellness, Entertainment) with increased data volume (2000 samples/class) and hyperparameter tuning (GridSearchCV).

Result: Accuracy improved drastically to 93.73%.

Top Performers: Support Vector Machine (SVM) and Multinomial Naive Bayes.

Future Scope
Implementation of Deep Learning models (e.g., LSTMs) to capture sequential context better.

Deployment of the best-performing model as a web application/API.
