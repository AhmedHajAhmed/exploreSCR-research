# Decoding Hate: Adapting Machine Learning Models for Hate Speech Detection in Levantine Arabic


**Ahmed Haj Ahmed**

Haverford College, Haverford, PA 

Email: ahajahmed@haverford.edu

## Abstract
Effective hate speech detection is crucial for creating safe online environments, especially in under-researched linguistic contexts like Levantine Arabic. This paper explores the challenges of hate speech detection in Levantine Arabic, focusing on the dialectal variations and cultural nuances that impact traditional natural language processing (NLP) tools. We evaluate the efficacy of various embedding methods and machine learning models in identifying hate speech, comparing custom-trained Word2Vec embeddings with standard models like GoogleNews Word2Vec and GloVe. Our results demonstrate the superiority of custom-trained embeddings and the effectiveness of simpler vectorization techniques like TF-IDF and Bag of Words. Additionally, we analyze the impact of data augmentation techniques, particularly oversampling, on model performance, highlighting its benefits in managing class imbalance. This research advances our understanding of hate speech detection in Levantine Arabic and underscores the importance of developing region-specific NLP tools for inclusive and safe online environments.

## Introduction
The rise of digital platforms has brought about a surge in online communication, making effective hate speech detection crucial for maintaining safe online spaces. However, detecting hate speech in linguistically diverse contexts presents significant challenges. Levantine Arabic, spoken across Syria, Palestine, Jordan, and Lebanon, exhibits considerable dialectal variation and is underrepresented in language technologies. Most existing NLP tools, designed for English, struggle to process the nuances of dialectal Arabic, hindering accurate hate speech detection.

This research aims to address these challenges by adapting machine learning models specifically for hate speech detection in Levantine Arabic. We investigate various embedding methods, including custom-trained Word2Vec, GoogleNews Word2Vec, and GloVe, comparing their efficacy with traditional vectorization techniques like TF-IDF and Bag of Words. Additionally, we explore the impact of data augmentation strategies, particularly focusing on oversampling techniques to manage class imbalance in training data.

By adapting machine learning models to the unique linguistic and cultural characteristics of Levantine Arabic, this research aims to contribute to the development of more inclusive and effective hate speech detection systems for underrepresented languages.


## Methodology
### Data Collection and Preprocessing
The dataset used in this research, titled "L-HSAB: A Levantine Twitter Dataset for Hate Speech and Abusive Language," provides a foundational corpus of Twitter posts annotated for hate speech, abusive language, and normal text. The dataset comprises Levantine Arabic tweets collected using the Twitter API, focusing on politically charged content from the timelines of politicians, social and political activists, and TV anchors. After manual filtering and annotation, the dataset was narrowed down to 5,846 tweets.

### Data Preprocessing Steps
The dataset underwent several preprocessing steps, including deduplication, outlier detection, and reclassification of sentiments. Text cleaning and normalization were performed to remove non-textual content and standardize Arabic characters. Data balancing was achieved through oversampling of the minority class ('hate') to address class imbalance.

### Embedding Techniques
Various embedding techniques were evaluated, including Bag of Words, TF-IDF, custom-trained Word2Vec, pre-trained Arabic fastText, and pre-trained GloVe. Custom-trained embeddings showed superior performance, highlighting the importance of domain-specific adaptations.

### Machine Learning Models
A diverse array of machine learning classifiers, including Logistic Regression, Random Forest, Support Vector Classifier (SVC), Decision Tree, K-Nearest Neighbors, XGBoost, and CatBoost, were employed to classify hate speech in Levantine Arabic.

## Results
### Analysis of Embedding Techniques
- Bag of Words and TF-IDF demonstrated strong performance, outperforming pre-trained embeddings like GoogleNews Word2Vec and GloVe.
- Custom-trained Word2Vec significantly outperformed generic embeddings, showcasing the importance of domain-specific adaptations.
- Pre-trained Arabic fastText performed well, benefiting from its ability to handle out-of-vocabulary words.

### Analysis of Data Imbalance
- Oversampling significantly improved classifier performance, outperforming both the original imbalanced dataset and the undersampled dataset.
- Classifiers trained on the oversampled dataset achieved higher F1 scores, indicating improved precision and recall.

## Conclusion
This research demonstrates the importance of adapting machine learning models and embedding techniques for hate speech detection in underrepresented languages like Levantine Arabic. Custom-trained embeddings and oversampling techniques show promise in improving classifier performance, highlighting the need for region-specific NLP tools to foster more inclusive and safe online environments.