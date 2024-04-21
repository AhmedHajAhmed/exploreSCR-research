import gensim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from train_and_evaluate import text_to_vector_GoogleNews_word2vec, text_to_vector_custom_word2vec, \
    text_to_vector_fasttext, text_to_vector_glove, load_glove_embeddings
import numpy as np
from main import df
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from gensim.models import KeyedVectors
import fasttext
from test_data_balance import df_undersampled, df_oversampled


def plot_roc_auc(df: pd.DataFrame) -> None:
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

    # Custom dataset-trained Word2Vec Vectorization
    tokenized_corpus = [gensim.utils.simple_preprocess(doc) for doc in df['Tweet']]
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
        'Logistic Regression': LogisticRegression(),
        # 'Multinomial NB': MultinomialNB(),
        'Random Forest': RandomForestClassifier(),
        'SVC': SVC(kernel='linear'),
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'XGBoost': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(verbose=0)
    }

    # Calculate the total number of subplots needed (one for each vectorization method)
    n_models = len(models)

    # Setting up the subplot grid
    n_cols = 3  # Adjust the number of columns as needed
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
    if n_models == 1:
        axs = [axs]  # Make axs iterable if there's only one subplot
    axs = axs.flatten()  # Flatten the grid for easy iteration

    for idx, (model_name, (X_train_vec, X_test_vec)) in enumerate(models.items()):
        ax = axs[idx]  # Current subplot
        ax.set_title(f'ROC Curves for {model_name}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal dashed line

        for classifier_name, classifier in classifiers.items():
            model = classifier.fit(X_train_vec, y_train)

            # Use predict_proba to get the probability scores, or decision_function for models like SVC
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test_vec)[:, 1]
            else:
                y_scores = model.decision_function(X_test_vec)

            # Compute ROC curve and ROC area for each class
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)

            # Plot the ROC curve for the classifier in the current subplot
            ax.plot(fpr, tpr, lw=2, label=f'{classifier_name} (AUC = {roc_auc:.2f})')

        ax.legend(loc="lower right")

    # Hide unused subplots if any
    for ax in axs[n_models:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()




# plot_roc_auc(df)
plot_roc_auc(df_undersampled)
# plot_roc_auc(df_oversampled)
