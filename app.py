from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle     
#importing pickle files
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def welcome():
    return render_template('fertilizer.html')
@app.route('/predict',methods=['POST'])
def predict():
    T = request.form.get('Temperature')
    H = request.form.get('Humidity')
    SM = request.form.get('soil_moisture')
    ST = request.form.get('soil_type')
    CT = request.form.get('crop_type')
    N = request.form.get('Nitrogen')
    K = request.form.get('Potassium')
    P = request.form.get('Phosporus')
    feature_list = [T, H, SM,ST,CT,N,K,P]
    single_pred = np.array(feature_list).reshape(1, -1)
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)
    crop_dict={1: "Urea", 2: "DAP", 3: "28-28", 4: "14-35-14", 5: "20-20", 6: "17-17-17", 7: "10-26-26"}
    
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('Fertilizer.html',result = result)
     
if __name__ == "__main__":
    app.run(debug=True)