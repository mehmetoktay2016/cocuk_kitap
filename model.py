#Kütüphanelerin yuklenmesi
import pickle #python nesnelerini kaydetmek ve cagirmak icin kullanilir.
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
#Model Tuning
import pickle
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error

import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)




df = pd.read_csv("data/kitap_cocuk.csv", index_col = 0)
#df = pd.read_csv("/Users/user/PycharmProjects/pythonProject5/12_hafta/kredi/churn_deployment/data/churn.csv", index_col = 0)
df = df.drop(["aciklama" , "kitap_isim"  , "yazari" , "yayinevi" ,"cevirmen" , "boyutlar_olcusu" ,"resim" ,"baski_sayisi" ,"oku_will","oku_ed","oku_ing"] , axis = 1)
df.head(2)

df= df.dropna()

#Bugunun tarihi
today_date = dt.datetime(2021 , 1 , 7)

#Cok onemli bir kod. Cunku tarihi istedigim formata getirdi
df['yayin_tarihi'] = pd.to_datetime(df['yayin_tarihi'], errors='coerce')

#Urun piyasaya cıkalı kac gun oldu ?
df["toplam_gun"] = (today_date - df.yayin_tarihi)

#days yazısının silinmesi
df.toplam_gun = df.toplam_gun.astype("string")
df.toplam_gun = df.toplam_gun.str.replace(" days" , "")
df.toplam_gun = df.toplam_gun.astype("int")

#yayin_tarihi değişkeninin dusurulmesi

df = df.drop(["yayin_tarihi"] , axis = 1)

#Birlestirme isleminde indeksler bozuldugu icin yenilememiz gerekmektedir.
df = df.reset_index()
df = df.drop(["index"] , axis = 1)
df.tail(2)

df.fiyati = df.fiyati.str.replace("," , ".")
df.fiyati = df.fiyati.astype("float")
df.kazanc = df.kazanc.str.replace("," , ".")
df.head(2)

df.satis_sayisi = df.satis_sayisi.str.replace(" adet satın alınmıştır." , "")
df.satis_sayisi = df.satis_sayisi.str.replace("Bu üründen " , "")
df.satis_sayisi = df.satis_sayisi.str.replace("." , "")


#Cok guzel bir kod
b  = [b  for b , x  in enumerate(df.satis_sayisi) if not any(c.isalpha() for c in x)]

df = df.iloc [b]
df = df.reset_index()
df = df.drop(["index"] , axis = 1)
len(df)

df.satis_sayisi = df.satis_sayisi.astype("string")
df.dtypes


a = [b  for b  in enumerate(df.satis_sayisi) if ("+" not in b)]
a = pd.DataFrame(a)
a = a.iloc[: , 0]
len(a)



df = df.loc[df.satis_sayisi != "0+" , ]
df = df.loc[df.satis_sayisi != "8+" , ]
df = df.loc[df.satis_sayisi != "8 +" , ]
df = df.loc[df.satis_sayisi != "9 +" , ]
df = df.loc[df.satis_sayisi != "9+" , ]
df = df.loc[df.satis_sayisi != "10+" , ]
df = df.loc[df.satis_sayisi != "0-7" , ]
df = df.loc[df.satis_sayisi != "0 +" , ]


df.satis_sayisi = df.satis_sayisi.astype("int")

df.head(2)
df.satis_sayisi.unique()


df.puan = df.puan.str.replace("Kazanacağınız Puan: " , "")
df.puan = df.puan.astype("int")

df = df.loc[df.sayfa_sayisi != "\n                                Hikaye                            " , ]
df.sayfa_sayisi = df.sayfa_sayisi.astype("int")

df.yorum_sayisi = df.yorum_sayisi * 1000
df.yorum_sayisi = df.yorum_sayisi.astype("str")
df.yorum_sayisi = df.yorum_sayisi.str.replace("000.0" , "")
df.yorum_sayisi = df.yorum_sayisi.astype("float")
df.kazanc = df.kazanc.astype("float")



X = df.drop('fiyati', axis=1)
y = df[["fiyati"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)





# LightGBM Model Tuning
lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01],
               "n_estimators": [200, 500],
               "max_depth": [6, 8, 10],
               "colsample_bytree": [1, 0.8,]}


lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_

# Final Model
lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X)


pickle.dump(lgbm_tuned, open("lrrr_model.pkl", 'wb'))

print("Model Kaydedildi")



