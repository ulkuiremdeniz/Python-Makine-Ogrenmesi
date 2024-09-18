'''
outlier : aykırı değer
KNN NEAREST NEIGHBORHOOD(K EN YAKIN KOMŞU)
k : kaç komşuya bakılacağını ifade eder
Bu algoritma, yeni bir veri noktası ile veri kümesindeki mevcut veri noktaları arasındaki mesafeyi hesaplayarak sınıflandırma veya tahmin yapar.
Öklidyen mesafesi kullanılır.
Sınıflandırma Problemi: Komşuların çoğunlukta olduğu sınıf, yeni veri noktası için tahmin edilir.
Regresyon Problemi: Komşuların hedef değişkenlerinin ortalaması alınarak yeni veri noktasının değeri tahmin edilir

Lazy learning :
bir öğrenme algoritmasının eğitim aşamasında herhangi bir model oluşturmadığı ve sadece veriyi saklayarak tahmin yapmak için veri noktaları arasındaki ilişkilere dayandığı bir yaklaşımdır
yavaş çalışır, bellek kullanımı yüksektir, eğitim maliyeti düşüktür.

Eager learning :
bir modelin eğitim aşamasında tüm veriyi kullanarak bir model öğrenmesi ve bu modeli tahmin yapmak için kullanması anlamına gelir.
hızlı tahmin, yüksek eğitim maliyeti
Veriler üzerinde bölgeleri oluşturmayı sağlar.
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

#KNN
from sklearn.neighbors import KNeighborsClassifier
#n_neighbors : komşu sayısı
#metric : hesaplama yöntemi
#k değerini arttırmak her zaman doğru bir yöntem değildir. Uyun veriye uygun k değeri vermelisin. k = 1 yaparak çıktıı tekrar kontrol et :)
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn_classifier.fit(X_train,y_train)
y_pred = knn_classifier.predict(X_test)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)