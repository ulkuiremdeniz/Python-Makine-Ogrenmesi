'''
MULTIPLE LINEAR REGRESSION :
birden fazla bağımsız değişkene bağlı olarak bağımlı değişkenin hesaplanması işlemidir.
y = ax1 + bx2 +cx3 +k +£
£:hata payı
'''
'''
KUKLA DEĞİŞKEN (DUMMY VARİABLE)
*Ön işleme aşamasında(kategorik verileri sayısala çevirirken) bir değişkenin başka değişkenle ifade edilmesidir.
*Makine öğrenmesinde her kolonun model üzerinde etkisi vardır.Yani aynı kolonun birden fazla tekrarı sonucu kendi yönünde çekmektedir.Bu yüzden
Dummy Veriable ile orijinal veriyi aynı anda kullanmamaya dikkat edilmelidir.Eğer bir kategorik değişkenden birden fazla dummy veriable ortaya çıkıyorsa 
ve bu dummy değişkenler birbirinden türetilebiliyorsa, yani biri diğerini ifade edebiliyorsa içlerinden biri seçilmelidir.
'''
'''
P-VALUE (OLASILIK DEĞERİ)
Bir hipotezin doğruluğunu hesaplamada kullanılır.
H0 : null hypothesis/sıfır hipotezi (başlangıçta ortaya atılan hipotez)
P-VALUE : bir hipotez testi sonucunda elde edilen test istatistiğinin, sıfır hipotezine dayanarak beklenenden ne kadar sapmış olduğuna dair bir olasılık değeridir.
p değerinin düşük olması (<0.05) : başlangıç hipotezinin hatalı olduğu kabul edilir.
p değerinin büyük olması (>0.05) : başlangıç hipotezinin doğru olduğu kabul edilir.
'''
'''
DEĞİŞKEN SEÇİMİ
Bağımsız değişkenlerden her biri bağımlı değişkeni aynı oranda mı etkiliyor, bağımsız değişkeni etkiliyor mu ?
Hangi değişkeni seçmeliyiz, hangileri bizim işimize yarar ?
    1.Bütün Değişkenleri Dahil Etmek
    2.Geriye Doğru Eleme (Backward Elimination)
    3.İleriye Seçim (Forward Selection)
    4.Çift Yönlü Eleme (Bidirection Elimination)
    5.Skor Karşılaştırması (Score Comparison)

P-değeri yüksek olan bir değişken, bağımlı değişken üzerinde anlamlı bir etkisi olmadığı anlamına gelir. Yani, bu değişken modelde tutulsa bile, modelin doğruluğunu veya performansını artırmaz.
P-değeri ne kadar yüksekse, o değişkenin modeldeki katkısı o kadar azdır.    
'''
'''
OLS :  bağımlı ve bağımsız değişkenler arasında doğrusal bir ilişki olduğunu varsayar ve bu ilişkinin katsayılarını tahmin etmek için kullanılır. 
Amaç, gözlemlenen değerler ile model tarafından tahmin edilen değerler arasındaki farkların karelerinin toplamını en aza indiren bir doğrusal denklem bulmaktır.
'''


#Kütüphaneler
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Veri Ön İşleme
dataset = pd.read_csv(r'C:\\Users\\PC\\Desktop\\Makine Öğrenmesi\\veriler.csv')
print(dataset)

#Cinsiyet kolonunun encode edilmesi (nümerikleştirme)
cinsiyet = dataset.iloc[:,-1:].values
print(cinsiyet)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cinsiyet[:,-1] = le.fit_transform(dataset.iloc[:,-1])
print(cinsiyet)

sonuc3 =pd.DataFrame(data = cinsiyet[:,:1],index=range(22),columns=['cinsiyet'])
print(sonuc3)

#ülke kolonunun encode edilmesi (nümerikleştirme)
ulke = dataset.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(dataset.iloc[:,0])
print(ulke)
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
#numpy dizilerinin dataframe dönüşümü
sonuc1 =pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])

yas = dataset.iloc[:,1:4].values
sonuc2 =pd.DataFrame(data=yas ,index=range(22),columns=['boy','kilo','yas'])
s = pd.concat([sonuc1,sonuc2],axis=1)
s2 =pd.concat([s,sonuc3],axis=1)
print(s2)

#Verilerin eğitim ve test veri setine bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

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
x = np.append(arr = np.ones((22,1)).astype(int),values=s,axis=1)
#tüm değişkenleri bir dizide tutarak eleminasyon yapılması amaçlanmaktadır.
x_liste = s.iloc[:,[0,1,2,3,4,5]].values
x_liste =np.array(x_liste,dtype=float)
model = sm.OLS(cinsiyet.astype(float),x_liste).fit()
print(model.summary())

#P değeri en büyük eleman elendi.
x_liste = s.iloc[:,[0,1,3,4,5]].values
x_liste =np.array(x_liste,dtype=float)
model = sm.OLS(cinsiyet.astype(float),x_liste).fit()
print(model.summary())

#P değeri en büyük eleman elendi.
x_liste = s.iloc[:,[0,1,3,4]].values
x_liste =np.array(x_liste,dtype=float)
model = sm.OLS(cinsiyet.astype(float),x_liste).fit()
print(model.summary())