import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import KeyedVectors
import fasttext
import numpy as np
from main import df, arabic_stop_words


def train_and_evaluate(df, arabic_stop_words):
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Class'], test_size=0.2, random_state=42)

    # BoW Vectorization
    vectorizer_bow = CountVectorizer(stop_words=arabic_stop_words)
    X_train_bow = vectorizer_bow.fit_transform(X_train)
    X_test_bow = vectorizer_bow.transform(X_test)

    # TF-IDF Vectorization
    vectorizer_tfidf = TfidfVectorizer(stop_words=arabic_stop_words)
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

    # Training and Evaluating Models
    models = {'BoW': (X_train_bow, X_test_bow),
              'TF-IDF': (X_train_tfidf, X_test_tfidf),
              'Pre-trained GoogleNews Word2Vec': (X_train_GoogleNews_w2v, X_test_GoogleNews_w2v),
              'Custom dataset-trained Word2Vec' : (X_train_custom_w2v, X_test_custom_w2v),
              'Pre-trained Arabic fastText': (X_train_fasttext, X_test_fasttext)}

    for name, (X_train_vec, X_test_vec) in models.items():
        model = LogisticRegression().fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        print(f"{name} Model Evaluation:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        # print("Classification Report:\n", classification_report(y_test, y_pred), "\n")
        print("===============================================")




def text_to_vector_GoogleNews_word2vec(text, model):
    word_vectors = [model[word] for word in text.split() if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)


def text_to_vector_custom_word2vec(text, model):
    words = gensim.utils.simple_preprocess(text)
    words = [word for word in words if word in model.wv.key_to_index]
    return np.mean([model.wv[word] for word in words], axis=0) if words else np.zeros(model.vector_size)


def text_to_vector_fasttext(text, model):
    word_vectors = [model.get_word_vector(word) for word in text.split()]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.get_dimension())




train_and_evaluate(df, arabic_stop_words)

