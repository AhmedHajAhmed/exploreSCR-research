import gensim
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from train_and_evaluate import text_to_vector_custom_word2vec
import numpy as np
from main import df
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from train_word2vec import combined_levantine_df, combined_arabic_df
from test_data_balance import df_undersampled, df_oversampled




def train_and_evaluate_csv(df: pd.DataFrame) -> pd.DataFrame:
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Class'], test_size=0.2, random_state=42)

    # Dataset-trained Word2Vec Vectorization
    tokenized_corpus_dataset = [gensim.utils.simple_preprocess(doc) for doc in df['Tweet']]
    word2vec_custom_model_dataset = Word2Vec(sentences=tokenized_corpus_dataset, vector_size=100, window=5, min_count=5, workers=4)
    X_train_custom_w2v = np.array([text_to_vector_custom_word2vec(text, word2vec_custom_model_dataset) for text in X_train])
    X_test_custom_w2v = np.array([text_to_vector_custom_word2vec(text, word2vec_custom_model_dataset) for text in X_test])

    # Levantine trained Word2Vec Vectorization
    tokenized_corpus_levantine = [gensim.utils.simple_preprocess(doc) for doc in combined_levantine_df['text']]
    word2vec_custom_levantine_model = Word2Vec(sentences=tokenized_corpus_levantine, vector_size=100, window=5, min_count=5, workers=4)
    X_train_custom_levantine_w2v = np.array([text_to_vector_custom_word2vec(text, word2vec_custom_levantine_model) for text in X_train])
    X_test_custom_levantine_w2v = np.array([text_to_vector_custom_word2vec(text, word2vec_custom_levantine_model) for text in X_test])

    # Arabic (Levantine and non-Levantine) trained Word2Vec Vectorization
    tokenized_corpus_levantine_and_nonlevantine = [gensim.utils.simple_preprocess(doc) for doc in combined_arabic_df['text']]
    word2vec_custom_model_levantine_and_nonlevantine = Word2Vec(sentences=tokenized_corpus_levantine_and_nonlevantine, vector_size=100, window=5, min_count=5, workers=4)
    X_train_custom_w2v_levantine_and_nonlevantine = np.array([text_to_vector_custom_word2vec(text, word2vec_custom_model_levantine_and_nonlevantine) for text in X_train])
    X_test_custom_w2v_levantine_and_nonlevantine = np.array([text_to_vector_custom_word2vec(text, word2vec_custom_model_levantine_and_nonlevantine) for text in X_test])

    # Word2Vec vectorizations
    models = {'Only dataset-trained Word2Vec': (X_train_custom_w2v, X_test_custom_w2v),
              'Custom Levantine dataset-trained Word2Vec': (X_train_custom_levantine_w2v, X_test_custom_levantine_w2v),
              'Custom Levantine + non-Levantine dataset-trained Word2Vec': (X_train_custom_w2v_levantine_and_nonlevantine, X_test_custom_w2v_levantine_and_nonlevantine)}

    # Classifiers to experiment with
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'SVC': SVC(kernel='linear'),
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'XGBoost': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(verbose=0)
    }

    results = []

    for classifier_name, classifier in classifiers.items():
        for model_name, (X_train_vec, X_test_vec) in models.items():
            try:
                model = classifier.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)
                # accuracy = accuracy_score(y_test, y_pred)
                accuracy = f1_score(y_test, y_pred)
                results.append({
                    'Classifier': classifier_name,
                    'Model': model_name,
                    'Accuracy': accuracy
                })
                print(results)
            except Exception as e:
                print(f"Error with {model_name} + {classifier_name}: {e}")
                results.append({
                    'Classifier': classifier_name,
                    'Model': model_name,
                    'Accuracy': 'Error'
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df



results_df = train_and_evaluate_csv(df)
# results_df = train_and_evaluate_csv(df_undersampled)
# results_df = train_and_evaluate_csv(df_oversampled)

pivoted_df = results_df.pivot(index='Model', columns='Classifier', values="Accuracy")

# Print the pivoted DataFrame
print(pivoted_df.to_string())



