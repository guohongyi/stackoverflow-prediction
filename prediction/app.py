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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi
import math
import os

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



def get_expert(predicted_tag,data_expert_id='./dataExpertID.csv', id_name='./expert_id_name.csv'):   
    if type(predicted_tag) == str:
        predicted_tag = [predicted_tag]
    id_name_csv = read_csv(id_name)
    experts_id = read_csv(data_expert_id)
    predicted_expert = np.empty([0])
    predicted_id = np.empty([0])
    # function for finding name from id
    def idToName(id):
        for row in id_name_csv:
            if row[0] == id:
                return row[1]
        return "noname"
    for i1 in range(len(predicted_tag)):
        idx = np.where(experts_id[:,0]==predicted_tag[i1])[0]
        predicted_id = np.append(predicted_id,experts_id[idx,1:])
    print(predicted_id)

    unique_id, ind = np.unique(predicted_id, return_inverse = True)
    #map names from ids
    unique_expert = list(map(idToName, unique_id))
    predicted_id = ", ".join(unique_id.tolist())  
    predicted_expert = ", ".join(unique_expert) #unique_expert already a list    
    return predicted_expert, predicted_id, unique_id

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
	
	
def toCsvString(arr):
    return ', '.join(arr)

def idToName(_id, id_name='./expert_id_name.csv'):
    id_name_csv = read_csv(id_name)
    for row in id_name_csv:
            if row[0] == _id:
                return row[1]
    return "noname"

def radar(unique_id, tags=np.array(['python','java','c','c++','javascript','r','mysql','css','html','php'])):
    file_name = './pagerank_top10.csv'
    pagerank0 = read_csv(file_name)
    #data0 = data0[data0[:,3].astype('int')>10,:]
    #tags = np.array(['python','java','c','c++','javascript','r','mysql','css','html','php'])
    pagerank1 = np.empty([0,pagerank0.shape[1]])
    for i0 in range(len(tags)):
        tmp=pagerank0[pagerank0[:,2]==tags[i0]]
        for i1 in range(unique_id.shape[0]):
        #unique_id=np.append(unique_id,tmp[np.argmax(tmp[:,3].astype('int')),1])
            pagerank1=np.append(pagerank1,tmp[tmp[:,0]==unique_id[i1]],axis=0)
    c1 = pagerank1[pagerank1[:,2]==tags[2]]
    c2 = pagerank1[pagerank1[:,2]==tags[3]]
    pagerank1 = pagerank1[pagerank1[:,2]!=tags[2]]
    pagerank1 = pagerank1[pagerank1[:,2]!=tags[3]]
    for i2 in range(c2.shape[0]):
        ind = np.where(c1[:,1]==c2[i2,1])[0]
        if len(ind)>0:
            c1[ind,3]=c1[ind,3].astype('int')+c2[i2,3].astype('int')
            c1[ind,4]=c1[ind,4].astype('int')+c2[i2,4].astype('int')
        else:
            c1 = np.vstack((c1,c2[i2,:]))
    c1[:,2]='c/c++'
    c1[:,5]=c1[:,3].astype('int')/c1[:,4].astype('int')
    pagerank1 = np.vstack((pagerank1,c1))
    tags = np.array(['python','java','c/c++','javascript','r','mysql','css','html','php'])
    #unique_users, indices = np.unique(data1[:,1],return_inverse = True)
    pagerank2 = np.tile(0.0,(unique_id.shape[0],9))
    for i3 in range(unique_id.shape[0]):#len(unique_id)
        #vars()[unique_id[i3]]=np.zeros([1,len(tags)])
        print(unique_id[i3])
        ind1 = np.where(pagerank1[:,0]==unique_id[i3])[0]
        tmp = pagerank1[ind1,:]
        for i4 in range(len(tags)):
            ind2 = np.where(tmp[:,2]==tags[i4])[0]
            if len(ind2)>0:
                pagerank2[i3,i4]=tmp[ind2,5]     
    # ------- PART 1: Create background
     
    # number of variable
    N = len(tags)
     
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
     
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
     
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
     
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], tags)
     
    # Draw ylabels
    ax.set_rlabel_position(0)
    tick = math.ceil(np.amax(pagerank2)/4)
    plt.yticks([0,tick,tick*2,tick*3], ["0","25","50","75"], color="grey", size=7)
    plt.ylim(-tick,tick*4)
    colors=['b','r','g','c','m','y','k','b','r','g','c','m','y','k']
    for i5 in range(pagerank2.shape[0]):
     
    # ------- PART 2: Add plots     
    # Plot each individual = each line of the data
        values = np.append(pagerank2[i5,:],pagerank2[i5,0])
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=idToName(unique_id[i5]))
        ax.fill(angles, values, colors[i5], alpha=0.1)    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig('./images/radar.png')
    return

def show_pie(prob):
    if len(prob) == 1:
        prob = prob[0]
    
    keys = ['0-1hr', '12-24hr', '1-3hr', '3-6hr', '6-12hr', '>24']
    new_order = [0,2,3,4,1,5]
    
    keys = np.array(keys)[new_order]
    prob = np.array(prob)[new_order]
    
    explode = [0, 0, 0, 0, 0, 0]  # only "explode" the 2nd slice (i.e. 'Hogs')
    explode[prob.argmax()] = 0.1
    
    fig1, ax1 = plt.subplots()
    ax1.pie(prob*100, explode=explode, labels=keys, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('./images/time_pie.png')
    
    return

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
    
    show_pie(prediction_time) #----------------- generate pie chart

    y_classes = prediction_time.argmax(axis=-1)

    source_time = encoder.inverse_transform(y_classes)



    source_tag = tag_encoder.inverse_transform(np.asarray([output]))

    print(source_time)
    print(source_tag[0][0])

    #########
    
    
    predicted_tag = source_tag[0]
    predicted_expert, predicted_id, unique_id = get_expert(predicted_tag)
    radar(unique_id) #---------------------------- generate radar plot
    
    data_path = './dataExpertID.csv'

    predicted_experts,expert_ids,unique_id = get_expert(predicted_tag, data_path)
    print(predicted_experts)
    print(predicted_tag)
    print(get_tag_image(predicted_tag))
    output = prediction_time
	

    return render_template('index.html', prediction_text='Time should be $ {}, Tag should be $ {}'.format(source_time, source_tag), prediction_time=source_time[0], prediction_tag=toCsvString(source_tag[0]), prediction_experts=predicted_experts, prediction_ids=expert_ids)

def get_tag_image(predicted_tag):
    if len(predicted_tag) == 1:
        predicted_tag = predicted_tag[0]
    if type(predicted_tag) == str:
        predicted_tag = [predicted_tag]
		
    img_path = './images'
    tag_images = []
    for tag in predicted_tag:
        tag_images.append(os.path.join(img_path, tag + '.png'))
    return tag_images

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
