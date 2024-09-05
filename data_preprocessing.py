'''
 CRISP-DM
    -veri madenciliği projeleri için kulanılan bir süreç modelidir.
    -altı ana fazdan oluşur
 1.İş Anlayışı(Business Understanding)
    -proje gereksinimlerinin oluşturulması ve çizelgelerin oluşturulması.
 2.Veri Anlayışı(Data Understanding)
    -veri toplama ve analiz etme
 3.Veri Hazırlığı(Data Preparation)
    -veri temizleme,ön işleme,eksik verilerin tamamlanması,verinin uygun formata getirilmesi
 4.Modelleme(Modeling)
    -farklı modelleme tekniklerini seçme ve uygulama
 5.Değerlendirme (Evaluation)
    -modellerin performansını ve doğruluğunu değerlendirme,modelin amacına uygun olduğunu değerlendirme
 6.Dağıtım (Deployment)
    -modelin kullanıma sunulması ve sonuçların incelenmesi,dökümantasyon


 data preprocessing : veri ön işleme
 DataFrame : verinin tablo şeklinde gösterimine verilen ad(iki boyutlu veri tablosu)
 CSV : Comma-separated values verileri saklamak ve taşımak için kullanılan dosya formatıdır, veriler virgüllerle ayrılır.


                    Nominal
        Kategorik <
                    Ordinal
 VERİ <
                    Oransal(Ratio)
        Sayısal <
                    Aralık(Interval)


Kategorik Veriler :Aralarında büyüklük küçüklük ilişkisi kurulamayan ,cebirsel işleme tabi tutulamayan verilerdir.


'''
'''
ONE-HOT ENCODING :
    -kategorik verileri sayısal verilere çevirmek için kullanılır.
    -her kategori için ayrı bir sutun oluşturur ve sadece ilgili kategorinin bulunduğu satıra 1 diğerlerine 0 değerini verir.
    -binary vektörlere dönüştürür
    -kategoriler arasında ilişki kurmaz
    -kategoriler arttıkça veri seti boyutu artar
    -sıralı olmayan kategoriler(nominal)
LABEL ENCODING:
    -kategorik verileri sayısal verilere çevirmek için kullanılır.
    -kategorik verileri doğrudan sayısal verilere çevirir.Her kategoriye bir tam sayı değeri atanır.
    -tam sayılara dönüştürür.
    -kategoriler arasında sıralı ilişki kurar
    -veri seti boyutunu arttırmaz
    -sıralı kategoriler(ordinal)
'''

#KÜTÜPHANELER
'''Pandas:veri manipülasyonu ve analizi için kullanılır.
   veri okuma,veri yazma,veri temizleme...
   '''
import pandas as pd
'''NumPy:bilimsel hesaplamalarda kullanılır.'''
import numpy as np
'''Matplotlib:veri görselleştirme ve grafik çizimi için araçlar sağlar'''
import matplotlib.pyplot as plt

#VERİ ÖN İŞLEME

#1.VERİ YÜKLEME
dataset = pd.read_csv(r"C:\\Users\\PC\Desktop\\Makine Öğrenmesi\\veriler.csv")
print(dataset)

#2.EKSİK VERİLER
missing_values = pd.read_csv(r"C:\\Users\\PC\Desktop\\Makine Öğrenmesi\\eksikveriler.csv")
print(missing_values)
'''veri setindeki eksik verileri(missing values) işlemek için kullanılan bir sınıftır.'''
from sklearn.impute import SimpleImputer
'''eksik verileri(nan) diğer verilerin ortalaması alınıp bu değerle doldurmak için bir imputer nesnesi oluşturuluyor'''
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
'''iloc : DataFrame'deki satır ve sutunları konum tabanlı seçmek için kullanılır.(tüm satırları ve sutun değeri 3 olan sutunu(yaş) seçtik)'''
yas = missing_values.iloc[:,3:4].values
print(yas)
#fit methoduyla yas kolonunun ortalaması hesaplandı
imputer = imputer.fit(yas)
#hesaplanan ortalama değerler non değerlerin yerine yazılıyor
yas = imputer.transform(yas)
print(yas)

'''Çoğu makine öğrenme algoritması sayısal veriler üzerinden çalışır'''
#3.KATEGORİK VERİLERİN SAYISAL VERİLERE DÖNÜŞTÜRÜLMESİ
ulke = dataset.iloc[:,0:1].values
print(ulke)
from sklearn import preprocessing
'''LabelEncoder : kategorik verileri sayısal değerlere dönüştürür'''
le = preprocessing.LabelEncoder()
'''ülkeler sayısal değerlere dönüştürülüyor'''
ulke[:,0] = le.fit_transform(dataset.iloc[:,0])
print(ulke)
'''OneHotEncoder : her kategori için ayrı bir sutun oluşturulur'''
ohe = preprocessing.OneHotEncoder()
#sayısal değerleri ikili bir vektöre dönüştürür
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#4.VERİ KÜMELERİNİN BİRLEŞTİRİLMESİ
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

boy_kilo = dataset.iloc[:,1:3].values
sonuc2 = pd.DataFrame(data=boy_kilo,index=range(22),columns=['boy','kilo'])

sonuc3 = pd.DataFrame(data =yas,index=range(22),columns=['yas'])
print(sonuc3)

cinsiyet = dataset.iloc[:,-1].values
sonuc4 = pd.DataFrame(data=cinsiyet ,index=range(22),columns=['cinsiyet'])

#axis = 1 : birleştirme işleminin sutun bazında yapılacağını belirtir
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
s2 = pd.concat([s,sonuc3],axis=1)
print(s2)
s3 = pd.concat([s2,sonuc4],axis=1)
print(s3)


'''Özellikler :ülke ,boy,kilo,yas bilgilerine bakılarak kişinin cinsiyet(hedef) tahmini yapılması amaçlanmaktadır.'''
#5.VERİLERİ KÜMESİNİN EĞİTİM VE TEST VERİLERİNE BÖLÜNMESİ
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(s2,sonuc4,test_size=0.33,random_state=0)


'''Makine Öğrenmesinde veri setindeki özniteliklerin farklı değer aralıklarını daha tutarlı bir aralığa getirme işlemidir.'''
#6.ÖZNİTELİK ÖLÇEKLEME
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)