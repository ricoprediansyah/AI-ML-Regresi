# ML-2 Regresi
# Sebelum mulai:

# Import Library

# Import library yang di buatuhkan
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('https://raw.githubusercontent.com/rasyidev/well-known-datasets/main/kc_house_3_features.csv')
data.head()

# Dataset memiliki 3 feature:

# sqft_living
# sqft_living15
# sqft_above
# dan sebuah label, yakni price.

# 1. Analisis Korelasi
# Lakukan analisis korelasi terhadap dataset.
data.corr()

# visualisasi menggnakan heatmap
plt.figure(figsize=(8,8))
sns.heatmap(data.corr(), annot=True)
plt.show()


# Hasil analisis korelasi menunjukkan bahwa:

# semua feature berkorelasi positif terhadap price
# feature sqft_living dan sqft_living15 memiliki koefisien korelasi tertinggi
# Split Dataset
# Split dataset menjadi:

# training data (X_train dan y_train) 80%
# testing data (X_test dan y_test) 20%

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print('Ukuran training dan testing data:')
print('Training data set:', X_train.shape, y_train.shape) #80% training data
print('Training data set:', X_test.shape, y_test.shape)   #20% testing data

# 2. Modeling
# Pada program ini, kita akan melatih lebih dari satu model regresi, antara lain:

# Regresi Linier
# Regresi Lasso
# Regresi Ridge
# Support Vector Regressor (SVR)
# Decision Tree Regressor (DTR)
# Latih semua model menggunakan training data (X_train dan y_train)

LinearReg = LinearRegression().fit(X_train,  y_train)
LassoReg = Lasso(alpha=0.1).fit(X_train,  y_train)
RidgeReg = Ridge(alpha=0.2).fit(X_train,  y_train)
SVReg = SVR().fit(X_train,  y_train)
DTReg = DecisionTreeRegressor(random_state=47).fit(X_train,  y_train)

# alpha=0.1 dan random_state=47 disebut sebagai hyperparameter.
# Umumnya, setiap algoritma AI memiliki hyperparameter yang bisa kita setel (tuning) sesuai keinginan.
# Setelan hyperparameter yang tepat mampu meningkatkan performa model.
# Kunjungi dokumentasi library untuk mempelajari tentang hyperparameter:

# Regresi Linier
# Regresi Lasso
# Regresi Ridge
# SVR
# DTR

# 3. Evaluasi Model Menggunakan Metric MSE dan R2
# Lakukan evaluasi pada semua model yang telah dilatih. Sebelum itu, kita perlu men-generate hasil prediksi tiap model.

# Generate:

# hasil prediksi training data (y pred train)
# hasil prediksi testing data (y pred test)
# menggunakan model.

#  Hasil prediksi model Regresi Linier
ypredtrain_reglin = LinearReg.predict(X_train)
ypredtest_reglin = LinearReg.predict(X_test)
# Hasil prediksi model Lasso
ypredtrain_lasso = LassoReg.predict(X_train)
ypredtest_lasso = LassoReg.predict(X_test)
# Hasil prediksi model Ridge
ypredtrain_ridge = RidgeReg.predict(X_train)
ypredtest_ridge = RidgeReg.predict(X_test)
# Hasil prediksi model SVR
ypredtrain_svr = SVReg.predict(X_train)
ypredtest_svr = SVReg.predict(X_test)
# Hasil prediksi model DTR
ypredtrain_dtr = DTReg.predict(X_train)
ypredtest_dtr = DTReg.predict(X_test)

# Evaluasi pertama dilakukan menggunakan metric MSE.

# Bandingkan:

# label training data (y_train) dengan hasil prediksi training data (y pred train)
# label testing data (y_test) dengan hasil prediksi testing data (y pred test)

# MSE model Regresi Linier
print('Nilai MSE data training Regresi Linier =', mean_squared_error(y_train, ypredtrain_reglin))
print('Nilai MSE data testing Regresi Linier =', mean_squared_error(y_test, ypredtest_reglin), '\n')
#MSE model Lasso
print('Nilai MSE data training Regresi Lasso =', mean_squared_error(y_train, ypredtrain_lasso))
print('Nilai MSE data testing Regresi Lasso =', mean_squared_error(y_test, ypredtest_lasso), '\n')
#MSE model Ridge
print('Nilai MSE data training Regresi Ridge=', mean_squared_error(y_train, ypredtrain_ridge))
print('Nilai MSE data testing Regresi Ridge =', mean_squared_error(y_test, ypredtest_ridge), '\n')
#MSE model SVR
print('Nilai MSE data training Regresi SVR =', mean_squared_error(y_train, ypredtrain_svr))
print('Nilai MSE data testing Regresi SVR =', mean_squared_error(y_test, ypredtest_svr), '\n')
#MSE model DTR
print('Nilai MSE data training Regresi DTR =', mean_squared_error(y_train, ypredtrain_dtr))
print('Nilai MSE data testing Regresi DTR =', mean_squared_error(y_test, ypredtest_dtr))

# Hasil evaluasi MSE menunjukkan bahwa:

# Model	MSE traing data	MSE testing data	Keputusan
# Regresi Linier	51040616225.03679	32881775262.15838	-
# Regresi Lasso	51040616225.0368	32881775271.07428	-
# Regresi Ridge	32881775271.07428	32881775254.902466	-
# SVR	113673471256.08711	37393496977.231895	Overfit
# DTR	455625000.0	40958648000.0	Goodfit
# Model DTR merupakan model terbaik karena MSE training dan testing tidak jauh berbeda.

# Note: Regresi Linier, Lasso, dan Ridge memiliki MSE training yang lebih tinggi dibanding MSE testing. Hal ini bisa terjadi karena data kita berjumlah sedikit. Salah satu solusinya adalah dengan menambah jumlah data dalam dataset.

# R^2 Score
print(f'R^2 score Regresi Linier: {LinearReg.score(X,Y)}')
print(f'R^2 score Regresi Lasso: {LassoReg.score(X,Y)}')
print(f'R^2 score Regresi Ridge: {RidgeReg.score(X,Y)}')
print(f'R^2 score SVR: {SVReg.score(X,Y)}')
print(f'R^2 score DT: {DTReg.score(X,Y)}')

# Pilih model dengan R^2 score mendekati 1.

# Hasil evaluasi R^2 score menunjukkan bahwa DTR merupakan model terbaik.

# Visualisasi Hasil Prediksi dengan Data Sebenarnya
# Visualisasikan perbandingan antara label testing data (y_test) dengan hasil prediksi testing data (y pred test) pada setiap model.

plt.plot(y_test.values)
plt.plot(ypredtest_reglin)
plt.title('Prediction vs Real Data Regresi Linier')
plt.xlabel('Data ke-')
plt.ylabel('Housing Price')
plt.legend(labels=['Prediction', 'Real'])
plt.show()

plt.plot(y_test.values)
plt.plot(ypredtest_lasso)
plt.title('Prediction vs Real Data Regresi Lasso')
plt.xlabel('Data ke-')
plt.ylabel('Housing Price')
plt.legend(labels=['Prediction', 'Real'])
plt.show()

plt.plot(y_test.values)
plt.plot(ypredtest_ridge)
plt.title('Prediction vs Real Data Regresi ridge')
plt.xlabel('Data ke-')
plt.ylabel('Housing Price')
plt.legend(labels=['Prediction', 'Real'])
plt.show()

plt.plot(y_test.values)
plt.plot(ypredtest_svr)
plt.title('Prediction vs Real Data Regresi svr')
plt.xlabel('Data ke-')
plt.ylabel('Housing Price')
plt.legend(labels=['Prediction', 'Real'])
plt.show()

plt.plot(y_test.values)
plt.plot(ypredtest_dtr)
plt.title('Prediction vs Real Data Regresi dtr')
plt.xlabel('Data ke-')
plt.ylabel('Housing Price')
plt.legend(labels=['Prediction', 'Real'])
plt.show()

# Kesimpulan
# Model DTR merupakan model terbaik berdasarkan hasil evaluasi menggunakan metrics MSE dan R^2.
# Model DTR selanjutnya kita pilih untuk memprediksi data baru.
# Model DTR juga sudah bisa kita deploy. Materi deployment akan disampaikan saat AI Domain.

# Input data baru
sqft_living = float(input('Input SQFT Living \t='))
sqft_living15 = float(input('Input SQFT Living 15 \t='))
sqft_above = float(input('Input SQFT Above\t='))
data_baru = [[sqft_living, sqft_living15, sqft_above]]

# Prediksi data baru menggunakan model DTR
hasil_prediksi = DTReg.predict(data_baru)
hasil_prediksi = float(hasil_prediksi)

# Cetak hasil prediksi (Profit)
print('\nPrediksi Housing Price', hasil_prediksi)
