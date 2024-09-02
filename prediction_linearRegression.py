#TAHMİN : PREDİCTİON
'''
*kategorik veriler üzerinde tahmin  : sınıflandırma(classification)
sayısal veriler üzerinde tahmin     : tahmin(prediction)
*öngörü : forecasting (örneklem uzayı dışındaki verilerin tahmin edilmesi)
*Regression : bağımsız değişkenler ile bağımlı değişken arasındaki ilişkiyi modelleyerek, bu ilişkinin yönünü ve büyüklüğünü belirlemeye ve gelecekteki gözlemler için tahminler yapmaya yardımcı olan bir tekniktir.
'''

''' Ay bilgisine bakılarak satış tahmini yapmak'''

#LINEAR REGRESSİON : DOĞRUSAL REGRESYON
#1.DATA PREPROCESSING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yükleme
dataset = pd.read_csv(r"C:\\Users\\PC\Desktop\\Makine Öğrenmesi\\satislar.csv")
print(dataset)

#veriyi parçalama
monts = dataset[['Aylar']]
print(monts)
sales = dataset[['Satislar']]
print(sales)

#veriyi test ve eğitim setine bölme
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(monts,sales,test_size=0.33,random_state=0)

#veriyi standartlaştırma (ortalama =0 ,standart sapma =1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

'''fit : modeli eğitmek için kullanılan method'''
#MODELİN İNŞA EDİLMESİ
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#TAHMİN
prediction = regressor.predict(x_test)
print(prediction)

#VERİ VE REGRESYON GÖRSELLEŞTİRME
#eğitim verilerini indexe göre sıraladık
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,regressor.predict(x_test))
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.show()