import pandas as pd
import openpyxl
from preprocessing_utils import preprocess_text


# Load DataFrames

df1 = pd.read_csv("data/word2vec_training_datasets/word2vec_training_datasets/hate_speech_mlma ar_dataset.csv")
df1 = df1[["tweet"]]

df2 = pd.read_csv("data/word2vec_training_datasets/word2vec_training_datasets/let-mi_train_part.csv")
df2 = df2[["text"]]

df3 = pd.read_table("data/word2vec_training_datasets/word2vec_training_datasets/L-HSAB.txt", delimiter="	")
df3 = df3[["Tweet"]]

df4 = pd.read_excel("word2vec_training_datasets/AJCommentsClassification-CF.xlsx")
df4 = df4[["body"]]

df5 = pd.read_excel("word2vec_training_datasets/LabeledDataset.xlsx")
df5 = df5[["commentText"]]

df6 = pd.read_excel("word2vec_training_datasets/TweetClassification-Summary.xlsx")
df6 = df6[["text"]]

df7 = pd.read_excel("word2vec_training_datasets/AJGT.xlsx")
df7 = df7[["Feed"]]

df8 = pd.read_csv("data/word2vec_training_datasets/word2vec_training_datasets/Tweets.txt", delimiter='\t', header=None, names=['text', 'label'])
df8 = df8[["text"]]


# Rename columns to 'text'
df1.rename(columns={"tweet": "text"}, inplace=True)
df3.rename(columns={"Tweet": "text"}, inplace=True)
df4.rename(columns={"body": "text"}, inplace=True)
df5.rename(columns={"commentText": "text"}, inplace=True)
df7.rename(columns={"Feed": "text"}, inplace=True)


# Concatenate all DataFrames vertically
combined_arabic_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
combined_levantine_df = pd.concat([df2, df3, df7], ignore_index=True)


# print("Number of rows in combined_arabic_df:", len(combined_arabic_df))
# print("Number of rows in combined_levantine_df:", len(combined_levantine_df))


# Remove duplicates
combined_arabic_df = combined_arabic_df.drop_duplicates(subset='text', keep='first')
combined_levantine_df = combined_levantine_df.drop_duplicates(subset='text', keep='first')

# print("Number of rows in combined_arabic_df after removing duplicates:", len(combined_arabic_df))
# print("Number of rows in combined_levantine_df after removing duplicates:", len(combined_levantine_df))


# Drop NaNs
combined_arabic_df.dropna(subset=['text'], inplace=True)
combined_levantine_df.dropna(subset=['text'], inplace=True)


# Apply the preprocess_text function
combined_arabic_df['text'] = combined_arabic_df['text'].apply(preprocess_text)
combined_levantine_df['text'] = combined_levantine_df['text'].apply(preprocess_text)


# print(combined_arabic_df)
# print(combined_levantine_df)


