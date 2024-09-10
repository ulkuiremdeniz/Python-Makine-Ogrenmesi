'''
RASSAL AĞAÇLAR (RANDOM FOREST)
-Ensemble Learning(Kollektif Öğrenme) : Birden fazla tahmin/sınıflandırma algoritmasının aynı anda kullanılarak daha başarılı sonuçlar elde edilmesi.
Veri kümesi küçük parçalara bölünerek her parçala farklı karar ağaçları oluşturulur ve ortalamaları hesaplanarak  çıktı belirlenir.(majority vote : çoğunluğun oyu)
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

from sklearn.ensemble import RandomForestRegressor
#n_estimators : kaç tane karar ağacı çizileceğini belirtiyor.
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Random Forest')
plt.show()