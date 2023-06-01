"Importing Required Library"
from flask import *
import glob
from pathlib import Path
import os
import pickle
from prediction import *
import numpy as np
import sys

''' Getting the latest model name '''
# list_of_files = glob.glob('.//models//*') 
# latest_file = max(list_of_files, key=os.path.getctime)
# filename = Path(latest_file)
# model = pickle.load(open(filename,'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    # print(int_features)
    number_point_need =  float(int_features[0])

    print("actual_value",number_point_need)
    
    # forecast = model.forecast(steps=12)
    memory_model = Model_Output()
    model_prediction = memory_model.prediction(number_point_need)
    print("------->",(model_prediction))
    # return str(model_prediction)
    # return model_prediction
    if int(model_prediction)>0:
        model_prediction = "Anamoly Detected"
    else:
        model_prediction = "Normal Usages"
    
    return render_template('index.html',prediction = model_prediction )



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''

    int_features = json.loads(request.get_data())
    print("Data ----> ",int_features)
    print("type(data) ----> ", type(int_features))

    Memory_Usage = int_features['Memory_Usage']#int(int_features.values()[0])
    final_features = [np.array([Memory_Usage])]
    
    prediction = Model_Output().prediction(final_features)
    print("-------------------------------------------------------------------------")
    print(prediction)
    print("##########################################")
    if int(prediction)>0:
        prediction = -1
    else:
        prediction = 1

    print(f"Model prediction: {prediction} ", file=sys.stdout)
    
    output = json.dumps({"Prediction":prediction})
    print(type(output))
    return output

""" ------------------------------------------------------------------------- """
''' Trying Integrating StreamLit'''
# @app.route('/stream_api',methods=['POST'])



# train = pd.read_csv("5_Minute_CPU_Usage_Data.csv")
# # train = train.rename(columns={"Well Name": "WELL"})


# @app.route("/api/data")
# def data():
#     selector = request.args.get("selector")
#     if not selector:
#         selector = "SHRIMPLIN"
#     # print(selector)
#     data = train[train["CPU_Usage"].isin([selector])]
#     # print(data)
#     return json.dumps(data.to_json())


# @app.route("/api/labels")
# def labels():
#     return json.dumps(train.WELL.unique().tolist())


''' End of Stream Lit code'''
""" ------------------------------------------------------------------------- """

if __name__=="__main__":
    app.run(debug=True)