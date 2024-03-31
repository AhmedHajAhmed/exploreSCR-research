from preprocessing_utils import *
from pyarabic import araby


# load datasets
path = "datasets/L-HSAB.txt"
df = pd.read_table(path, delimiter="	")


# Remove duplicated tweets based on the 'Tweet' column, keeping the first occurrence
df = df.drop_duplicates(subset='Tweet', keep='first')


# IQR method for identifying outliers

# Calculate the length of each tweet
df['Tweet_Length'] = df['Tweet'].apply(len)

# Calculate the first and third quartile
Q1 = df['Tweet_Length'].quantile(0.25)
Q3 = df['Tweet_Length'].quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# Define the bounds for non-outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Tweet_Length'] < lower_bound) | (df['Tweet_Length'] > upper_bound)]


# Remove outliers

# Filter the DataFrame to remove outliers
df = df[(df['Tweet_Length'] >= lower_bound) & (df['Tweet_Length'] <= upper_bound)]

# Replacing 'Abusive' with 'Hate' in the 'Class' column
df['Class'] = df['Class'].replace({'abusive': 'hate'})

# Apply the preprocess_text function to each tweet in the 'Tweet' column
df['Tweet'] = df['Tweet'].apply(preprocess_text)


# Apply tokenization to each tweet in the 'Tweet' column
# df['Tokenized_Tweet'] = df['Tweet'].apply(araby.tokenize)


# Apply the mapping to the 'Class' column
df['Class'] = df['Class'].map({'hate': 1, 'normal': 0})


# Define an empty list to store the stop words
arabic_stop_words = []
# Open the text file and read each line in the file
with open('arabic_stopwords_list.txt', 'r', encoding='utf-8') as file:
    for line in file:
        arabic_stop_words.append(preprocess_text(line.strip()))






# print(df)

