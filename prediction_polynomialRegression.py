'''
POLYNOMIAL REGRESSION
    y = ax^n +bx^k+... +cx^t +k +£
    y = ax1^n +bx2^k+... +cx2^t +£
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

#linear regression ile tehmin yapmış olsaydık sonuç ne olurdu ?
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.title('LINEAR Regression')
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.show()

#POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly =poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.title('POLYNOMIAL Regression (degree: 2)')
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()

#4. dereceden bir regression modeli oluştursaydık ?
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly =poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.title('POLYNOMIAL Regression (degree: 4)')
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()