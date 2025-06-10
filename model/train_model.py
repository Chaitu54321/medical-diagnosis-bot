import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import ast

# Load the cleaned dataset
df = pd.read_csv("dispred dataset//cleaned_dataset.csv")



# Convert stringified lists to real lists
df['all_symptoms'] = df['all_symptoms'].apply(ast.literal_eval)

# Get all unique symptoms
all_symptoms = sorted(list(set(symptom for symptoms in df['all_symptoms'] for symptom in symptoms)))


for symptom in all_symptoms:
    df[symptom] = df['all_symptoms'].apply(lambda x: 1 if symptom in x else 0)

# Now your X and y
X = df[all_symptoms]  # these are now 0/1 features
y = df['Disease']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))


import joblib
joblib.dump(all_symptoms, 'symptom_list.pkl')  # Save the features used


joblib.dump(model, 'disease_prediction_model.pkl')




