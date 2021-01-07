import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle  # python nesnelerini kaydetmek ve cagirmak icin kullanilir.
from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np


from pycaret.anomaly import predict_model

import numpy as np
from flask import Flask, request, render_template
import pickle  # python nesnelerini kaydetmek ve cagirmak icin kullanilir.

app = Flask(__name__)  # flask uygulamasını initialize eder. baslatir.
model = pickle.load(open('lrrr_model.pkl', 'rb'))  # kayitli modeli cagirma


# flask root (kök dizin) için anasayfanın tanımlanması
@app.route('/')
def home():
    return render_template('template.html')  # anasayfa icin hazirlanmis olan template belirtilir.


# yukaridaki bolum sonrasi template.html icindeki hareketleri okumaliyiz.

@app.route('/predict', methods=['POST'])
def predict():
    # Arayüzdeki text'lere girilen değerleri alıp hesaplama sonrası tekrar arayüze göndereceğiz.
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    predicted_y = int(np.round(prediction, 2))



    # html çıktısına geri gondermek
    return render_template('template.html', prediction_text='Kitap Fiyatı : {} TL olmalı.'.format(predicted_y))



@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
