import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast

from bs4 import BeautifulSoup
import lxml
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

import h5py
import csv



app = Flask(__name__)
tag_encoder = MultiLabelBinarizer()
tag_model = tf.keras.models.load_model('model-tag-new.h5',custom_objects={'GlorotUniform': tf.keras.initializers.glorot_uniform()})
time_model = tf.keras.models.load_model('model_Time.h5',custom_objects={'GlorotUniform': tf.keras.initializers.glorot_uniform()})

nltk.download('stopwords')
nltk.download('wordnet')

data_tag = pd.read_csv('dataTag.csv')
token = ToktokTokenizer()
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('tokenizer-time.pickle', 'rb') as handle:
    tokenizertime = pickle.load(handle)


data_tag['tags'] = data_tag['tags'].apply(ast.literal_eval)




def read_csv(file_name, my_delim=',', my_quote='"'):
    len_csv = 0
    file_content = []
    with open(file_name, 'rU') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=my_delim, quotechar=my_quote)
        
        counter = 0
        for row in spamreader:
            #print ', '.join(row)
            if len_csv == 0:
                len_csv = len(row)
            elif len_csv != len(row):
                print('[warning] row %i size not equal %i/%i' % (counter, len(row), len_csv))
                print(', '.join(row))
            
            clean_row = []
            for list_item in row:
                if len(list_item) > 0 and list_item[-1] in [' ']:
                    list_item = list_item[:-1]
                if ',' in list_item:
                    list_item=list_item.replace(',',' ')
                #list_item=float(list_item)
                clean_row.append(list_item)
            file_content.append(clean_row)
            counter += 1        
        csv_np = np.array(file_content)    
    #file_contnet_pd = pd.read_csv(file_path)
    return csv_np

def get_expert(predicted_tag,data_expert='./dataExpert.csv',data_expert_id='./dataExpertID.csv'):    
    experts = read_csv(data_expert)
    experts_id = read_csv(data_expert_id)
    predicted_expert = np.empty([0])
    predicted_id = np.empty([0])
    for i1 in range(len(predicted_tag)):
        idx = np.where(experts[:,0]==predicted_tag[i1])[0]
        predicted_expert = np.append(predicted_expert,experts[idx,1:])
        predicted_id = np.append(predicted_id,experts_id[idx,1:])
    unique_expert, ind = np.unique(predicted_expert, return_inverse = True)
    unique_id, ind = np.unique(predicted_id, return_inverse = True)
    predicted_expert = ", ".join(unique_expert.tolist())
    predicted_id = ", ".join(unique_id.tolist())  
    return predicted_expert, predicted_id

def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']

def removeStopWords(text):
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))

def removePunctuation(text):
    punct = '!"$%&\'()*,./:;<=>?@[\\]^_`{|}~'
    words = token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    for w in words:
        punctuation_filtered.append(regex.sub('', w))

    filtered_list = strip_list_noempty(punctuation_filtered)

    return ' '.join(map(str, filtered_list))

def lemmatizeWords(text):
    words = token.tokenize(text)
    listLemma = []
    for w in words:
        x = lemma.lemmatize(w, pos="v")
        listLemma.append(x.lower())
    return ' '.join(map(str, listLemma))

@app.route('/')
def home():
    # loadTokens()
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [str(x) for x in request.form.values()]

    question = features[1]
    title = features[0]

    # questions = [question].apply(lambda x: BeautifulSoup(x, "lxml").get_text())

    question = BeautifulSoup(question, "lxml").get_text()



    # Remove stopwords, punctuation and lemmatize for text in body
    # question = question.apply(lambda x: removeStopWords(x))
    # question = question.apply(lambda x: removePunctuation(x))
    # question = question.apply(lambda x: lemmatizeWords(x))

    question = removeStopWords(question)
    question = removePunctuation(question)
    question = lemmatizeWords(question)

    # # Remove stopwords, punctuation and lemmatize for title. Also weight title 3 times
    title = removeStopWords(title)
    title = removePunctuation(title)
    title = lemmatizeWords(title)
    title =  title+' '+title+' '+title


    feature = [title + ' ' + question]

    # final_features = [np.array(int_features)]
    # feature = ["r convert igraph visnetwork r convert igraph visnetwork r convert igraph visnetwork i find way convert igraph visnetwork refer interactive arules arulesviz visnetwork suppose conversion igraph visnetwork result show convert visnetwork result different i try demonstrate issue use sample data data groceries library arules #pre-defined library library arules library arulesviz library visnetwork library igraph #get sample data amp get association rule data groceries rule visedges arrows visoptions highlightnearest t plot top 10 association rule via use visnetwork note for visnetwork diagram size intercept node indicate lift higher lift larger size intercept node unlike igraph diagram size intercept node indicate support colour intercept node indicate lift let compare igraph visnetwork by refer association rule table format rule no10 rule smallest lift suppose size intercept node smallest end smallest problems i try drill igdf"]
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

    print(source_tag[0][0])

    #########
    
    
    predicted_tag = source_tag[0]
    data_path = './dataExpert.csv'

    predicted_experts,expert_ids = get_expert(predicted_tag, data_path)
    print(predicted_experts)
    print(expert_ids)
    output = prediction_time
    
    # expert[source_tag[0][0]]
    return render_template('index.html', 
                           prediction_text='Time should be $ {}, Tag should be $ {}, Recommended experts are {}, their IDs are {}'.format(
                               source_time, source_tag, predicted_experts, expert_ids
                            ))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
