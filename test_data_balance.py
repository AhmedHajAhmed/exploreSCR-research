from main import df
import pandas as pd
from sklearn.utils import resample


# Separate data into two different DataFrames based on class labels
df_majority = df[df['Class'] == 0]      # 0 = normal
df_minority = df[df['Class'] == 1]      # 1 = hate

# Perform undersampling of the majority class
df_majority_undersampled = resample(df_majority,
                                    replace=False,    # Don't replace the samples (no duplication)
                                    n_samples=len(df_minority),  # Set the number of samples to match the minority class
                                    random_state=123) # Set a random state for reproducibility of results

# Combine the undersampled majority class DataFrame with the original minority class DataFrame
df_undersampled = pd.concat([df_majority_undersampled, df_minority])

# Shuffle the combined DataFrame to ensure that the data points are well mixed
df_undersampled = df_undersampled.sample(frac=1, random_state=123).reset_index(drop=True)

# print(df_undersampled)

# print(df_undersampled['Class'].value_counts())





############################################################################################################




# Separate data into two different DataFrames based on class labels
df_majority = df[df['Class'] == 0]      # 0 = normal
df_minority = df[df['Class'] == 1]      # 1 = hate

# Perform oversampling of the minority class
df_minority_oversampled = resample(df_minority,
                                   replace=True,     # Enable replacement to duplicate the samples
                                   n_samples=len(df_majority),  # Set the number of samples to match the majority class
                                   random_state=123) # Set a random state for reproducibility of results

# Combine the oversampled minority class DataFrame with the original majority class DataFrame
df_oversampled = pd.concat([df_majority, df_minority_oversampled])

# Shuffle the combined DataFrame to ensure that the data points are well mixed
df_oversampled = df_oversampled.sample(frac=1, random_state=123).reset_index(drop=True)

# Print the resulting DataFrame (optional)
# print(df_oversampled)

# Print the counts of each class to verify balancing
# print(df_oversampled['Class'].value_counts())


print(df_oversampled['Class'].value_counts())
