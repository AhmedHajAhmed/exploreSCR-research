import string
import re
from typing import Callable, Dict, List
from pyarabic.araby import strip_tashkeel, strip_tatweel
import pandas as pd

def remove_punctuations(text: str) -> str:
    # This function removes Arabic & English punctuations from a string
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def normalize_arabic(text: str) -> str:
    # This function normalizes Arabic text by standardizing variations of Alif, Ya, Hamza, Taa Marbuta, and Kaf characters
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_emoji(text: str) -> str:
    "This function removes emojis from a string"
    emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text


def remove_nums(text: str) -> str:
    "This function removes Arabic and English numbers from a string"
    text = re.sub('[\u0661-\u0669]+', '', text)          # remove Arabic numbers
    text = ''.join(i for i in text if not i.isdigit())   # remove English numbers
    return text


def remove_english_letters(text: str) -> str:
    # This function replace any English letters with an empty string
    return re.sub('[a-zA-Z]', '', text)


def count_preprocessed_tweets(df: pd.DataFrame, preprocessing_functions: List[Callable[[str], str]]) -> Dict[str, int]:
    # This function counts the number of tweets in a DataFrame that are affected by each preprocessing function
    # A tweet is considered 'affected' by a preprocessing function if the function alters the tweet's content

    counts = {}  # Dictionary to store counts for each preprocessing function

    for function in preprocessing_functions:
        count = 0
        for text in df['Tweet']:
            if text != function(text):
                count += 1
        counts[function.__name__] = count  # Store count with function's name as key

    return counts


def preprocess_text(text: str) -> str:
    # This function applies a series of text preprocessing steps to the input text
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    text = remove_punctuations(text)
    text = normalize_arabic(text)
    text = remove_emoji(text)
    text = remove_nums(text)
    text = remove_english_letters(text)
    return text


