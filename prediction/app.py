import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast
import h5py



app = Flask(__name__)
tag_encoder = MultiLabelBinarizer()
tag_model = tf.keras.models.load_model('model-tag-new.h5',custom_objects={'GlorotUniform': tf.keras.initializers.glorot_uniform()})
time_model = tf.keras.models.load_model('model_Time.h5',custom_objects={'GlorotUniform': tf.keras.initializers.glorot_uniform()})

data_tag = pd.read_csv('dataTag.csv')


# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('tokenizer-time.pickle', 'rb') as handle:
    tokenizertime = pickle.load(handle)


data_tag['tags'] = data_tag['tags'].apply(ast.literal_eval)



@app.route('/')
def home():
    # loadTokens()
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]


    # final_features = [np.array(int_features)]
    feature = ["r convert igraph visnetwork r convert igraph visnetwork r convert igraph visnetwork i find way convert igraph visnetwork refer interactive arules arulesviz visnetwork suppose conversion igraph visnetwork result show convert visnetwork result different i try demonstrate issue use sample data data groceries library arules #pre-defined library library arules library arulesviz library visnetwork library igraph #get sample data amp get association rule data groceries rule visedges arrows visoptions highlightnearest t plot top 10 association rule via use visnetwork note for visnetwork diagram size intercept node indicate lift higher lift larger size intercept node unlike igraph diagram size intercept node indicate support colour intercept node indicate lift let compare igraph visnetwork by refer association rule table format rule no10 rule smallest lift suppose size intercept node smallest end smallest problems i try drill igdf"]
    #feature = ["sdk version issue failed uploading build sdk version issue failed uploading build sdk version issue failed uploading build warning itms-90725 sdk version issue this app build ios 120 sdk starting march 2019 ios apps submit app store must build ios 121 sdk later include xcode 101 laterbut im use xcode 101 sdk version 101 show screen shoot sdk version description 1 deploymenttargey2 httpsistackimgurcomupqbcpng httpsistackimgurcombznqgpng"]


    tag_encoder = MultiLabelBinarizer()


    bag_of_words_test = tokenizer.texts_to_matrix(feature)

    prediction_tag = tag_model.predict(bag_of_words_test)

    bag_of_words_test = tokenizertime.texts_to_matrix(feature)

    prediction_time = time_model.predict(bag_of_words_test)

    print(prediction_tag )



    encoder = LabelEncoder()
    data_time = pd.read_csv('dataTime.csv')
    encoder.fit_transform(data_time['lbl'])

    tag_encoder.fit_transform(data_tag['tags'])
    #


    output = prediction_tag[0].copy()
    output[output >= 0.2] = 1
    output[output < 0.2] = 0

    y_classes = prediction_time.argmax(axis=-1)

    source_time = encoder.inverse_transform(y_classes)



    source_tag = tag_encoder.inverse_transform(np.asarray([output]))

    print(source_time)

    print(source_tag)

#########


    output = prediction_time

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)