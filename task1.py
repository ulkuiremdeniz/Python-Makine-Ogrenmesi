#Kütüphanler
import pandas as pd
import numpy as np

#Veri yükleme
dataset = pd.read_csv(r"C:\\Users\\PC\Desktop\\Makine Öğrenmesi\\task1.csv")
print(dataset)

#Kategorik verileri sayısal verilere dönüştürme
outlook = dataset.iloc[:,0:1].values
print(outlook)

windy = dataset.iloc[:,3:4].values
print(windy)

play = dataset.iloc[:,-1:].values
print(play)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
windy2 = le.fit_transform(dataset.iloc[:,3])
play2 = le.fit_transform(dataset.iloc[:,4])

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

print(outlook)
print(windy2)
print(play2)

'''
tüm sutunları encode edebilmek için :
veriler = dataset.apply(preprocessing.LabelEncoder().fit_transform)
print(veriler)
'''


#Veri kümelerinin birleştirilmesi
sonuc = pd.DataFrame(data=outlook,index=range(14),columns=['Overcast','Rainy','Sunny'])
sonuc2 = pd.DataFrame(data=play2,index=range(14),columns=['Play'])
sonuc3 = pd.DataFrame(data=windy2,index=range(14),columns=['Windy'])

s = pd.concat([sonuc,dataset.iloc[:,1:2]],axis=1)
s2 = pd.concat([s,sonuc3],axis=1)
s3 = pd.concat([s2,sonuc2],axis=1)

print(s3)

humidity = dataset.iloc[:,2:3]

#Verilerin eğitim ve test veri setine bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s3,humidity,test_size=0.33,random_state=0)

#Model eğitimi
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)

#Geriye Doğru Eleme (Backward Elimination)
#En yüksek P değerine sahip olan bağımsız değişkeni ele
import statsmodels.api as sm
#diziye sabit değişken ekleniyor(1).
x = np.append(arr = np.ones((14,1)).astype(int),values=s,axis=1)
#tüm değişkenleri bir dizide tutarak eleminasyon yapılması amaçlanmaktadır.
x_liste = s3.iloc[:,[0,1,2,3,4,5]].values
x_liste =np.array(x_liste,dtype=float)
model = sm.OLS(humidity.astype(float),x_liste).fit()
print(model.summary())

#P değeri en büyük eleman elendi.
x_liste = s3.iloc[:,[0,1,2,3,5]].values
x_liste =np.array(x_liste,dtype=float)
model = sm.OLS(humidity.astype(float),x_liste).fit()
print(model.summary())

#modelin yeniden eğitilmesi
x = x_train.iloc[:,:4]
x2 = x_train.iloc[:,5:]
x_train = pd.concat([x,x2],axis=1)
x = x_test.iloc[:,:4]
x2 = x_test.iloc[:,5:]
x_test = pd.concat([x,x2],axis=1)

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)