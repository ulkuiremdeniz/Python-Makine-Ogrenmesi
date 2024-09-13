'''
CLASSIFICATION :
Kategorik verilerin tahminine sınıflandırma denir.

LOGISTIC FUNCTION :
özellikle iki sınıflı (binary) sınıflandırma problemlerinde kullanılan istatistiksel bir modeldir.
model, bağımlı değişkenin iki olası sonuçtan birine ait olma olasılığını tahmin eder.(0-1, evet-hayır, başarılı-başarısız)
Lojistik regresyonun temelinde sigmoid fonksiyonu bulunur.

h(t) = 1 / (1 + e^-t)
t = aX + b

'''
'''
CONFUSION MATRIX (KARMAŞIKLIK MATRİSİ)
Sınıflandırma problemlerinde modelin başarısını değerlendirmek için kullanılan bir tablodur.

                        Gerçek Pozitif          Gerçek Negatif
 
Tahmin Pozitif          True Positive           False Positive   

Tahmin Negative         False Negative          True Negative

True Positive (TP): Modelin pozitif olarak doğru tahmin ettiği örnekler (Gerçek pozitif ve tahmin de pozitif).
True Negative (TN): Modelin negatif olarak doğru tahmin ettiği örnekler (Gerçek negatif ve tahmin de negatif).
False Positive (FP): Modelin pozitif olarak yanlış tahmin ettiği örnekler (Gerçek negatif ama tahmin pozitif). Bu, Type I Error olarak da bilinir.
False Negative (FN): Modelin negatif olarak yanlış tahmin ettiği örnekler (Gerçek pozitif ama tahmin negatif). Bu da Type II Error olarak adlandırılır.

Accuracy : modelin tüm tahminlerinin ne kadar doğru olduğunu gösterir.(%)
diagonal (köşegendeki) veriler başarılı sınıflandırmayı belirtir.Köşegen dışındakiler başarısız tahminleri belirtir.
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

'''fit: eğitmek / transform: öğrenilen bilgiyi uygulamak'''
#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(x_train)
X_test = sc_x.transform(x_test)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression(random_state=0)
log_classifier.fit(X_train,y_train)

y_pred = log_classifier.predict(x_test)
print(x_test)
print(y_test)
print(y_pred)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)