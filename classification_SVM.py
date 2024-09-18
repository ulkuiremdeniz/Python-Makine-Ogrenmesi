'''
SUPPORT VECTOR REGRESSION (DESTEK VEKTÖR REGRESYONU)
Amaç : Sınıfları birbirinden ayıran en iyi fonksiyonu bulmak.İki sınıf arasındaki margin değerleri en uzak olan fonksiyonu seçmek.
Veriler destek vektörünün dışında olmalı.
Veriler algoritmaya girmeden önce ölçeklendirilmesi gerekli.
Sınıf ayırımlarının bir fonksiyonla ayırılabileceğini düşünen bir algoritmadır.
'''

#Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\PC\Desktop\Makine Öğrenmesi\veriler.csv')
print(dataset)

x = dataset.iloc[:,1:4].values
y = dataset.iloc[:,-1:].values

#verilerin eğitim ve test veri setine bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)


#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(x_train)
X_test = sc_x.transform(x_test)

#SVM
from sklearn.svm import SVC
#kernel : makine öğreniminde, verileri daha yüksek boyutlu bir uzaya haritalayarak doğrusal olmayan problemlerin çözülmesini sağlayan bir fonksiyondur.Türleri; linear, polynomial, rbf, sigmoid
svc_classifier = SVC(kernel='rbf')
svc_classifier.fit(X_train,y_train)
y_pred = svc_classifier.predict(X_test)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)