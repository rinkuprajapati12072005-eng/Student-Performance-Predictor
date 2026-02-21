import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

print("🚀 Training Student Performance Model...")

data = {
    'age': [18,19,17,20,16,18,19,17,18,19,20,17,18,19,16]*4,
    'study_hours': [7,5,8,3,9,6,4,10,7,5,2,8,6,4,9]*4,
    'absences': [2,5,0,10,1,3,7,0,2,6,12,1,4,8,0]*4,
    'parent_education': ['high school','bachelor','some college','high school','master','bachelor','high school','master','some college','bachelor','high school','master','bachelor','high school','master']*4,
    'grade': ['Pass','Pass','Pass','Fail','Pass','Pass','Fail','Pass','Pass','Pass','Fail','Pass','Pass','Fail','Pass']*4
}

df = pd.DataFrame(data)
le = LabelEncoder()
df['parent_encoded'] = le.fit_transform(df['parent_education'])
X = df[['age','study_hours','absences','parent_encoded']]
y = (df['grade'] == 'Pass').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.2%}")

with open('student_model.pkl', 'wb') as f:
    pickle.dump((model, le), f)
print("🎉 Model saved as student_model.pkl")
