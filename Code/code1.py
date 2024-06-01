import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
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

#Memasukan Dataset InSDN
df = pd.read_csv('combined.csv')
df.head()

#Melakukan Perubahan Pada Label data kelas Target
df['Label'].value_counts()
df['Label'] = df['Label'].replace({'DDoS': 'DDoS', 'DDoS ': 'DDoS'})
df.replace({'Label': {'BFA': 1, 'DDoS': 1, 'DoS': 1, 'Probe': 1, 
                      'Web-Attack': 1, 'BOTNET':1, 'Normal':0, 'U2R':1}}
                      , inplace=True)

#Memeriksa Dataset
df.describe()
df.info()

#Memberishkan Nilai Null
def check_null_values(df):
    null_values = df.isnull().sum()
    if null_values.sum() == 0:
        print("Dataset tidak memiliki naili Null/kosong.")
    else:
        print("Nilai null sebanyak:")
        print(null_values[null_values > 0])
check_null_values(df)

#Membuang Fitur tidak penting secara manual
columns_to_drop = ['Flow ID']
df.drop(columns_to_drop, axis=1, inplace=True)
columns_to_drop = ['Src IP']
df.drop(columns_to_drop, axis=1, inplace=True)
columns_to_drop = ['Dst IP']
df.drop(columns_to_drop, axis=1, inplace=True)
columns_to_drop = ['Timestamp']
df.drop(columns_to_drop, axis=1, inplace=True)
df.head()

#Melakukan Noramalisasi Data
X = df.drop(['Label'], axis = 1)
y = df['Label']
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("\nSetelah Normalisasi")
print(X.to_string(index=False))

#Melakukan Selection Feture
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFlod(5), scoring='accuracy')
rfecv.fit(X, y)
print('Jumlah Fitur Optimal:{}'.format(rfecv.n_features_))

# Visualisasi hasil RFECV
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], linestyle='-', color='teal')
plt.xlabel('Number of Features Selected')
plt.ylabel('Mean Test Score')
plt.show()

#Visualisasikan Peringkat Fitur
rfecv.estimator_.feature_importances_
df.drop(df.columns[np.where(rfecv.support_ == False)[0]], axis =1, inplace=True)
X = df.drop(['Label'], axis =1)
dset['attr'] = X.columns
dset['importance'] = rfecv.estimator_.feature_importances_
dset = dset.sort_values(by='importance', ascending =True)
plt.figure(figsize=(14, 6))
plt.barh(y=dset['attr'], width=dset['importance'], color='teal')
plt.title('RFECV -FITUR TERPENTING', fontsize =20, fontweight='bold', pad =10)
plt.xlabel('Fitur', fontsize=14, labelpad=10)
plt.show()

# Simpan hasil subset feature ke file CSV baru
df.to_csv('Fitur Terpilih.csv', index=False)