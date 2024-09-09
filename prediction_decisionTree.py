'''
KARAR AĞACI (DECISION TREE)
-Sınıflandırma ve tahminde kullanılır.
-Karar ağacı, veriyi dallara ayırarak her bir dalda bir karar verir
ve en sonunda sınıflandırma veya tahmin yapmak için bir yaprak düğümüne (sonuç düğümüne) ulaşır.
-Karar ağacındaki her düğüm, bir özelliğe dayalı olarak veriyi bölerek çalışır.
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

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Decision Tree Regression')
plt.show()