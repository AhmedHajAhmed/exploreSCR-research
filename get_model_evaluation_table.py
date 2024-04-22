import gensim
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from train_and_evaluate import text_to_vector_GoogleNews_word2vec, text_to_vector_custom_word2vec, text_to_vector_fasttext, text_to_vector_glove, load_glove_embeddings
import numpy as np
from main import df
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from gensim.models import KeyedVectors
import fasttext
from train_and_evaluate import combined_df
from test_data_balance import df_undersampled, df_oversampled




def train_and_evaluate_csv(df: pd.DataFrame) -> pd.DataFrame:
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Class'], test_size=0.2, random_state=42)

    # BoW Vectorization
    vectorizer_bow = CountVectorizer()
    X_train_bow = vectorizer_bow.fit_transform(X_train)
    X_test_bow = vectorizer_bow.transform(X_test)

    # TF-IDF Vectorization
    vectorizer_tfidf = TfidfVectorizer()
    X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
    X_test_tfidf = vectorizer_tfidf.transform(X_test)

    # Pre-trained GoogleNews Word2Vec Vectorization
    GoogleNews_word2vec_model_path = 'models/GoogleNews-vectors-negative300.bin'
    GoogleNews_word2vec_model = KeyedVectors.load_word2vec_format(GoogleNews_word2vec_model_path, binary=True)
    X_train_GoogleNews_w2v = np.array([text_to_vector_GoogleNews_word2vec(text, GoogleNews_word2vec_model) for text in X_train])
    X_test_GoogleNews_w2v = np.array([text_to_vector_GoogleNews_word2vec(text, GoogleNews_word2vec_model) for text in X_test])

    # dataset-trained Word2Vec Vectorization
    # tokenized_corpus = [gensim.utils.simple_preprocess(doc) for doc in df['Tweet']]
    tokenized_corpus = [gensim.utils.simple_preprocess(doc) for doc in combined_df['text']] ######
    word2vec_custom_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=5, workers=4)
    X_train_custom_w2v = np.array([text_to_vector_custom_word2vec(text, word2vec_custom_model) for text in X_train])
    X_test_custom_w2v = np.array([text_to_vector_custom_word2vec(text, word2vec_custom_model) for text in X_test])


    # Pre-trained Arabic fastText Vectorization
    fasttext_model_path = 'models/cc.ar.300.bin'
    fasttext_model = fasttext.load_model(fasttext_model_path)
    X_train_fasttext = np.array([text_to_vector_fasttext(text, fasttext_model) for text in X_train])
    X_test_fasttext = np.array([text_to_vector_fasttext(text, fasttext_model) for text in X_test])

    # Pre-trained GloVe Vectorization
    glove_path = 'models/glove/glove.6B.300d.txt'
    glove_embeddings = load_glove_embeddings(glove_path)
    X_train_glove = np.array([text_to_vector_glove(text, glove_embeddings) for text in X_train])
    X_test_glove = np.array([text_to_vector_glove(text, glove_embeddings) for text in X_test])

    models = {'BoW': (X_train_bow, X_test_bow),
              'TF-IDF': (X_train_tfidf, X_test_tfidf),
              'Pre-trained GoogleNews Word2Vec': (X_train_GoogleNews_w2v, X_test_GoogleNews_w2v),
              'Custom dataset-trained Word2Vec': (X_train_custom_w2v, X_test_custom_w2v),
              'Pre-trained Arabic fastText': (X_train_fasttext, X_test_fasttext),
              'Pre-trained GloVe': (X_train_glove, X_test_glove)}

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        # 'Multinomial NB': MultinomialNB(),
        'Random Forest': RandomForestClassifier(),
        'SVC': SVC(kernel='linear'),
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'XGBoost': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(verbose=0)
    }
    # Initialize results dictionary
    results = []

    for classifier_name, classifier in classifiers.items():
        for model_name, (X_train_vec, X_test_vec) in models.items():
            try:
                model = classifier.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)
                # accuracy = accuracy_score(y_test, y_pred) ####
                accuracy = f1_score(y_test, y_pred)
                results.append({
                    'Classifier': classifier_name,
                    'Model': model_name,
                    'Accuracy': accuracy
                })
            except Exception as e:
                print(f"Error with {model_name} + {classifier_name}: {e}")
                results.append({
                    'Classifier': classifier_name,
                    'Model': model_name,
                    'Accuracy': 'Error'
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Optionally print the DataFrame
    return results_df



# results_df = train_and_evaluate_csv(df)
# results_df = train_and_evaluate_csv(df_undersampled)
results_df = train_and_evaluate_csv(df_oversampled)


# Assuming results_df is your DataFrame from the previous output
pivoted_df = results_df.pivot(index='Model', columns='Classifier', values="Accuracy")

# Optionally save the pivoted DataFrame to a CSV
pivoted_df.to_csv('model_evaluation.csv')

# Print the pivoted DataFrame
print(pivoted_df.to_string())




