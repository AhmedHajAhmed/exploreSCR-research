from preprocessing_utils import preprocess_text


# Define an empty list to store the stop words
arabic_stop_words = []

# Open the text file and read each line in the file
with open('data/arabic_stopwords_list.txt', 'r', encoding='utf-8') as file:
    for line in file:
        arabic_stop_words.append(preprocess_text(line.strip()))

custom_arabic_stop_words = ["يا", "شو", "عم", "مش", "شي", "انو", "مين", "هيك", "هيدا", "اللي", "انك", "منك",
                            "وانت", "كنت", "رح", "عنا", "ليش", "فيك", "هلق", "نحنا", "مو", "انتو"]

arabic_stop_words = arabic_stop_words + custom_arabic_stop_words

