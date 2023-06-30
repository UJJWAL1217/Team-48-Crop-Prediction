# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
import pickle
# #
# # Load the csv file
# df = pd.read_csv("Crop_recommendation.csv")
# #
# print(df.head())
#
# X = df[["N", "P", "K", "temperature","humidity","ph","rainfall"]]
# y = df["label"]
#
# # Split the dataset into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
#
# # Feature scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test= sc.transform(X_test)
#
# # Instantiate the model
# classifier = RandomForestClassifier()
#
# # Fit the model
# classifier.fit(X_train, y_train)
#
# # Make pickle file of our model
# pickle.dump(classifier, open("model.pkl", "wb"))

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')  # Replace 'crop_data.csv' with your dataset file

# Split the data into features and labels
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
# predictions = model.predict(X_test)

pickle.dump(model, open("model.pkl", "wb"))
# Evaluate the model
# accuracy = model.score(X_test, y_test)
# print("Accuracy:", accuracy)

# Example usage: Predict crop for a new set of features
# new_features = [[117 ,32,34,26.2724184,52.12739421,6.758792552,127.1752928,]]  # Replace with your own set of features
# predicted_crop = model.predict(new_features)
# print("Predicted crop:", predicted_crop)

