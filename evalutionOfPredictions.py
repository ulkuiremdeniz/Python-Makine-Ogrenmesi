'''
TAHMİN ALGORİTMALARININ DEĞERLENDİRİLMESİ (EVALUTION OF PREDICTIONS)
    R-SQUARE : 1- [ Topla(gerçek_değer - tahmin)^2 / Topla(gerçek_değer - ort_tahmin)^2 ]
    R-SQUARE : [hata karelerinin toplamı] / [ortalama farkların toplamı]
    Herhangi bir tahmin algoritmasının başarı değerini ölçebileceğimiz sayısal bir değer verir.
    r-square 'in 1 e yaklaşması algoritmanın başarı oranının yüksek olduğunu ifade eder.

    ADJUSTED R SQUARE : modeli iyileştirmek için eklenen yeni değişkenler sisteme asla negatif bir etki yapmaz.
        r-square değerini değiştirmez bu da yeni eklenen değişkenin sisteme ne kadar olumlu veya olumsuz etki yaptığını görmemizi engeller.
        Bu problemden kurtulmak için ADJUSTED R SQUARE yöntemi kullanılır.
        1-(1-R-SQUARE)*[n-1 / n-p-1]
        n : değişken sayısı
        p : eklenen değişken sayısı
'''

#KÜTÜPHANELER
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

#VERİ ÖN İŞLEME
dataset = pd.read_csv(r'C:\Users\PC\Desktop\Makine Öğrenmesi\maaslar.csv')
print(dataset)

#data frame dilimleme (slice)
egitim_seviyesi = dataset.iloc[:,1:2]
maas = dataset.iloc[:,2:]

#NumPy array dönüşümü
x = egitim_seviyesi.values
y = maas.values

#Random Forest
from sklearn.ensemble import RandomForestRegressor
#n_estimators : kaç tane karar ağacı çizileceğini belirtiyor.
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)
print("Random Forest Regression R2 Değeri")
print(r2_score(y,regressor.predict(x)))

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)
print("Decision Tree Regression R2 Değeri")
print(r2_score(y,regressor.predict(x)))

#Suppot Vector Regression
#Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

print("Suppot Vector Regression R2 Değeri")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))


#POLYNOMIAL REGRESSION
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly =poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

print("Polynomial Regression R2 Değeri")
print(r2_score(y,lin_reg2.predict(poly_reg.fit_transform(x))))