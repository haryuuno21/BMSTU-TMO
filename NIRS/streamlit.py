import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("students.csv")

df_model = df.drop(columns=["Student_ID"])
# ordinal encoding
grade_mapping = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
df_model["Final_Grade_Ordinal"] = df_model["Final_Grade"].map(grade_mapping)

X = df_model.drop(columns=["Final_Grade"])
y = df_model["Final_Grade_Ordinal"]

# one-hot encoding
categorical_features = ["Gender", "Preferred_Learning_Style", "Participation_in_Discussions", 
                        "Use_of_Educational_Tech", "Self_Reported_Stress_Level"]

X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.title("Модель KNN для классификации")

n_neighbors = st.slider("Выберите число соседей (k)", min_value=1, max_value=15, value=3)
metric = st.selectbox("Выберите метрику расстояния", options=['euclidean', 'manhattan', 'minkowski'])

model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.write(f"Точность модели при k = {n_neighbors} и метрики {metric}: **{acc:.2f}**")