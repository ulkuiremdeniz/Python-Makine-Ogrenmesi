'''
SUPPORT VECTOR REGRESSION(DESTEK VEKTÖR REGRESYONU)
Sınıflandırma: Sınıflandırma yaparken max margin aralığını sağlayan doğruyu seçmeyi sağlar.
Regression: Bir margin aralığında maximum veri noktasını bulmayı amaçlar.Çizilen doğrular,eğriler üzerinde min margin aralığında max veriyi elde etmek.
SVR nin regresyondaki amacı min margin aralığında max veriyi elde edebilmeyi sağlayan fonksiyonu bulmaktır.
SVR kullanılırken veriler ölçeklendirilmelidir.
'''

#KÜTÜPHANELER
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#VERİ ÖN İŞLEME
dataset = pd.read_csv(r'C:\Users\PC\Desktop\Makine Öğrenmesi\maaslar.csv')
print(dataset)

#data frame dilimleme (slice)
egitim_seviyesi = dataset.iloc[:,1:2]
maas = dataset.iloc[:,2:]

#NumPy array dönüşümü
x = egitim_seviyesi.values
y = maas.values

#Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

#Model eğitimi
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

#Tahmin sonuçlarını görselleştirme
plt.title('SVR Regression')
plt.scatter(x_olcekli, y_olcekli, color='red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color='blue')
plt.show()