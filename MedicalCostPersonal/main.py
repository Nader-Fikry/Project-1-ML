import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load data from spreed sheet.
dataset = pd.read_csv("insurance.csv")
input_data = dataset.iloc[:, : -1].values
output_data = dataset.iloc[:, 6].values

# Encode three columns
encoder = LabelEncoder()
input_data[:, 1] = encoder.fit_transform(input_data[:, 1])
input_data[:, 4] = encoder.fit_transform(input_data[:, 4])
input_data[:, 5] = encoder.fit_transform(input_data[:, 5])


# Split data into train data & test data.
input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.2)
