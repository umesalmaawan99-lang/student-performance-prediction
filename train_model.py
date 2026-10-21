import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
data = pd.read_csv('student_data.csv')

# 2. Define features (X) and target (y)
X = data[['study_hours', 'attendance_percentage', 'previous_grades', 'extracurricular_activities', 'sleep_hours']]
y = data['passed']

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. Save the trained model to a file
joblib.dump(model, 'student_model.pkl')
print("Model saved successfully as 'student_model.pkl'!")