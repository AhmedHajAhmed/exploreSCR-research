import gensim
import numpy as np


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


def text_to_vector_glove(text, embeddings):
    words = text.split()
    word_vectors = [embeddings[word] for word in words if word in embeddings]

    if not word_vectors:
        return np.zeros(300)  # Assuming using 300-dimensional GloVe vectors

    return np.mean(word_vectors, axis=0)


def load_glove_embeddings(path):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings
