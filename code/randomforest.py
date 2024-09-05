import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')
# Read the data
df = pd.read_csv("breast-cancer-wisconsin-data_data 2.csv")

# Define the features as a list of column names
features = ['radius_mean', 'perimeter_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 
            'concavity_mean', 'symmetry_mean', 'perimeter_se', 'concave points_se', 
            'fractal_dimension_se', 'area_worst', 'fractal_dimension_worst']

# Extract features (X) and target variable (y)
X = df.loc[:, features]

# Encode the target variable to numeric values
le = LabelEncoder()
y = le.fit_transform(df['diagnosis'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create and fit the RandomForestRegressor model
reg = RandomForestRegressor(n_estimators=100, random_state=0)
reg.fit(X_train, y_train)

# Make predictions and calculate the score
predictions = reg.predict(X_test[0:10])
score = reg.score(X_test, y_test)

print("Model Score:", score)

# Visualize the first tree in the Random Forest
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
tree.plot_tree(reg.estimators_[0],
               feature_names=features, 
               filled=True)
plt.show()

def predict_cancer():
    print("Enter the following details to predict the diagnosis (0: Benign, 1: Malignant):")

    # Taking input from the user for each feature
    radius_mean = float(input("radius_mean: "))
    perimeter_mean = float(input("perimeter_mean: "))
    texture_mean = float(input("texture_mean: "))
    area_mean = float(input("area_mean: "))
    smoothness_mean = float(input("smoothness_mean: "))
    concavity_mean = float(input("concavity_mean: "))
    symmetry_mean = float(input("symmetry_mean: "))
    perimeter_se = float(input("perimeter_se: "))
    concavepoints_se = float(input("concave points_se: "))
    fractal_dimension_se = float(input("fractal_dimension_se: "))
    area_worst = float(input("area_worst: "))
    fractal_dimension_worst = float(input("fractal_dimension_worst: "))

    # Create a NumPy array from the input
    new_patient = np.array([[radius_mean, perimeter_mean, texture_mean, area_mean, smoothness_mean, 
                             concavity_mean, symmetry_mean, perimeter_se, concavepoints_se, 
                             fractal_dimension_se, area_worst, fractal_dimension_worst]])

    # Predict the class of the new patient
    predicted_class = reg.predict(new_patient)

    # Output the prediction (0 for benign, 1 for malignant)
    print(f"The predicted diagnosis for the input cancer is: {int(predicted_class[0])} (0: Benign, 1: Malignant)")

# Call the function to classify a patient
predict_cancer()
