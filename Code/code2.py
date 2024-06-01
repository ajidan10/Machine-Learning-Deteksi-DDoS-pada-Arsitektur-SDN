import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFoldKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('Fitur_Terpilih.csv')
df

df['Label'].value_counts()
df.describe()
df.info()

X = df.drop(['Label'], axis = 1)
y = df['Label']

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("\nSetelah Normalisasi")
print(X.to_string(index=False))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print('data X train',X_train.shape)
print('data X test',X_test.shape)
print('data y train',y_train.shape)
print('data y test',y_test.shape)

# Original class distribution plot
unique_classes, class_counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.bar(unique_classes, class_counts, color='skyblue')
plt.title('Distribusi kelas Data Train')
plt.xlabel('Kelas Target')
plt.ylabel('Jumalah data kelas')

# SMOTE-resampled class distribution plot
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled, color='skyblue')
plt.title('Distribusi Kelas Setelah SMOTE')
plt.xlabel('Kelas Target')
plt.ylabel('Jumlah Data Kelas')

plt.tight_layout()  # Ensures proper spacing between subplots
plt.show()

print('data y train',X_train.shape)
print('data y train',y_train.shape)
print('data y train',X_resampled.shape)
print('data y train',y_resampled.shape)

# Inisialisasi dan latih model Decision Tree
dt_model = DecisionTreeClassifier(max_depth= 4, min_samples_split=6, random_state=420)
dt_model.fit(X_resampled, y_resampled)
# Lakukan prediksi pada data uji
y_pred_dt = dt_model.predict(X_test)
# Evaluasi model
accuracy = accuracy_score(y_test, y_pred_dt)
print("Accuracy:", accuracy)
# Lihat laporan klasifikasi
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Inisialisasi dan latih model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, min_samples_split=4, random_state=42)
rf_model.fit(X_resampled, y_resampled)
# Lakukan prediksi pada data uji
y_pred_rf = rf_model.predict(X_test)
# Evaluasi model
accuracy = accuracy_score(y_test, y_pred_rf)
print("Accuracy:", accuracy)
# Lihat laporan klasifikasi
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Inisialisasi model Naive Bayes
model = GaussianNB()
# Latih model
model.fit(X_resampled, y_resampled)
# Prediksi pada set pengujian
y_pred = model.predict(X_test)
# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# Tampilkan laporan klasifikasi
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)
# Train the classifier on the training set
svm_classifier.fit(X_resampled, y_resampled)
# Make predictions on the testing set
y_pred_svm = svm_classifier.predict(X_test)
# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy: {accuracy * 100:.2f}%")
# Tampilkan laporan klasifikasi
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Inisialisasi model MLP untuk klasifikasi
model = MLPClassifier(hidden_layer_sizes=(10, ), max_iter=1000, random_state=42)
# Latih model
model.fit(X_resampled, y_resampled)
# Prediksi pada set pengujian
y_pred = model.predict(X_test)
# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# Tampilkan laporan klasifikasi
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Hasil akurasi untuk masing-masing model
model_names = ['DT', 'RF', 'NB', 'SVM', 'MLP']
accuracies = [0.9655, 0.9996, 0.5689, 0.9269, 0.9663]  # Gantilah dengan nilai akurasi yang sesuai
# Warna yang berbeda untuk setiap model
colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange', 'lightpink']
# Visualisasikan akurasi dengan warna yang berbeda
plt.figure(figsize=(6, 6))
bars = plt.bar(model_names, accuracies, color=colors)
plt.title('Akurasi Model Machine Learning')
plt.xlabel('Model')
plt.ylabel('Akurasi')
plt.ylim(0, 1)  # Set batas y-axis antara 0 dan 1
plt.show()

from joblib import dump, load
# Save data to a file using Joblib
dump(rf_model, 'model2.joblib')
# Load data from the Joblib file
loaded_data_joblib = load('model2.joblib')
# Print the loaded data
print("Loaded Data (Joblib):", loaded_data_joblib)