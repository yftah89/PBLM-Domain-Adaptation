import xml.etree.ElementTree as ET
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
import os
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional,TimeDistributed
import pickle
from keras.preprocessing.text import Tokenizer
import re
import pre
from keras.models import load_model
import numpy as np
from keras.models import load_model
from keras.models import Model
from keras.layers import Merge
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
np.random.seed(7)


#fix some encoding issues
def getClear_full(sentence):
    r = re.findall(r'\b\w+\b', sentence.lower())

    r = " ".join(r)
    r = (r.decode('unicode_escape').encode('ascii', 'ignore'))
    return r

def getClearList(sentences):
    clearSentences = []
    for s in sentences:
        clearSentences.append(getClear_full(s))
    return  clearSentences

def XML2arrayRAW(neg_path, pos_path):
    reviews = []
    negReviews = []
    posReviews = []

    neg_tree = ET.parse(neg_path)
    neg_root = neg_tree.getroot()
    for rev in neg_root.iter('review'):
        reviews.append(rev.text)
        negReviews.append(rev.text)



    pos_tree = ET.parse(pos_path)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review'):
        reviews.append(rev.text)
        posReviews.append(rev.text)

    return reviews,negReviews,posReviews

def GetTopNMI(n,CountVectorizer,X,target):
    MI = []
    length = X.shape[1]


    for i in range(length):
        temp=mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs,MI


def getCounts(X,i):

    return (sum(X[:,i]))

def extract_and_split(neg_path, pos_path):
    reviews,n,p = XML2arrayRAW(neg_path, pos_path)
    #train, train_target, test, test_target = split_data_balanced(reviews,1000,200)
    train=reviews
    train_target=[]
    test = []
    test_target=[]
    train_target = [0]*1000+[1]*1000
    return train, train_target, test, test_target

#keras has issues with connecting CNN on top of masked LSTM - https://github.com/keras-team/keras/issues/7588
#In order to deal with connecting CNN to the LSTM input, I first run the LSTM, save its outputs and then feed it to the
#CNN network
def partial(model, x):
    batch_size = len(x)/10
    init = model.predict_on_batch(x[0:batch_size])
    for i in range (1,10):
        temp = model.predict_on_batch(x[i*batch_size:(i+1)*batch_size])
        init = np.append(init, np.atleast_3d(temp), axis=0)
    return  init


def PBLM_CNN(src,dest,pivot_num,max_review_len,embedding_vecor_length_rep,topWords,hidden_units_num_rep,
             filters, kernel_size):
    model_path = src+"_to_"+dest+"/models/model_"+src+"_"+dest+"_"+str(pivot_num) + "_" + str(
        hidden_units_num_rep) + "_" +str(embedding_vecor_length_rep)+ "_" + ".model"
    model = load_model(model_path)
    split_dir = src + "_to_" + dest
    # gets all the train and test for sentiment classification
    with open(split_dir + "/split/train", 'rb') as f:
        train = pickle.load(f)
    with open(split_dir + "/split/test", 'rb') as f:
        val = pickle.load(f)


    unlabeled, source, target = pre.XML2arrayRAW("data/" + src + "/" + src + "UN.txt",
                                                 "data/" + dest + "/" + dest + "UN.txt")

    dest_test, source, target = XML2arrayRAW("data/" + dest + "/negative.parsed",
                                             "data/" + dest + "/positive.parsed")
    unlabeled = getClearList(unlabeled)
    train = getClearList(train)
    tok = Tokenizer(nb_words=topWords, split=" ")
    tok.fit_on_texts(train + unlabeled)
    train_count = 800
    X_train = tok.texts_to_sequences(train)
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_len)
    Y_train = [0] * train_count + [1] * train_count
    val = getClearList(val)
    X_val = tok.texts_to_sequences(val)
    X_val = sequence.pad_sequences(X_val, maxlen=max_review_len)
    val_count = 200
    Y_val =  [0] * val_count + [1] * val_count
    dest_test = getClearList(dest_test)
    X_test = tok.texts_to_sequences(dest_test)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_len)
    test_count = 1000
    Y_test = [0]*test_count+[1]*test_count
    #loading the PBLM model without the softmax layer
    modelT = Sequential()
    for i in range(len(model.layers)-1):
        modelT.add(model.layers[i])
        modelT.layers[i].trainable = False
        modelT.layers[i].mask_zero = False
    modelT.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print modelT.summary()



    #getting the input vectors, for more information read the "partial" function comments
    X_test =partial(modelT, X_test)
    X_train =  partial(modelT, X_train)
    X_val =  partial(modelT, X_val)
    #reshaping the input for the CNN network
    X_train = X_train.reshape(X_train.shape[0], max_review_len, hidden_units_num_rep)
    X_test = X_test.reshape(X_test.shape[0], max_review_len, hidden_units_num_rep)
    X_train = X_train.reshape(X_train.shape[0], max_review_len, hidden_units_num_rep)



    train_data = X_train
    val_data = X_val
    test_data = X_test
    sent_model = Sequential()
    sent_model.add(Conv1D(filters, kernel_size, border_mode='valid', activation='relu', input_shape=(max_review_len, hidden_units_num_rep)))
    # we use max pooling:
    sent_model.add(GlobalMaxPooling1D())
    sent_model.add(Dense(1, activation='sigmoid'))
    sent_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print sent_model.layers
    print(sent_model.summary())

    model_str = src + "_to_" + dest + "/sent_models_cnn/model_" + str(pivot_num)  +"_" + str(hidden_units_num_rep)+"_.model"
    filename = model_str
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    #stops as soon as the validation loss stops decreasing
    modelCheckpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True,
                                      save_weights_only=False, mode='min', period=1)
    # saving only the best model
    earlyStopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')
    sent_model.fit(train_data, Y_train, validation_data=(val_data, Y_val), nb_epoch=10, batch_size=16,callbacks=[earlyStopping,modelCheckpoint])
    print(sent_model.summary())
    print sent_model.get_config()
    sent_model = load_model(filename)
    val_score, val_acc =sent_model.evaluate(val_data, Y_val, batch_size=16)
    print('val loss:', val_score)
    print('val accuracy:', val_acc)
    test_score, test_acc = sent_model.evaluate(test_data, Y_test, batch_size=16)
    print('Test loss:', test_score)
    print('Test accuracy:', test_acc)

    score_path = src+"_to_"+dest+"/results/cnn/results.txt"
    sentence = "pivots = " + str(pivot_num) + " HU rep " + str(
        hidden_units_num_rep) + " word rep size " + str(embedding_vecor_length_rep)  +  " the val acc " + str(val_acc) + " test acc "+str(test_acc)

    if not os.path.exists(os.path.dirname(score_path)):
        os.makedirs(os.path.dirname(score_path))

    with open(score_path , "a") as myfile:
        myfile.write(sentence+"\n")


def PBLM_LSTM(src,dest,pivot_num,max_review_len,embedding_vecor_length_rep,topWords,hidden_units_num_rep, hidden_units_num):


    model_path = src+"_to_"+dest+"/models/model_"+src+"_"+dest+"_"+str(pivot_num) + "_" + str(
        hidden_units_num_rep) + "_" +str(embedding_vecor_length_rep)+ "_" + ".model"
    model = load_model(model_path)
    split_dir =  src + "_to_" + dest
    # gets all the train and test for sentiment classification
    with open(split_dir + "/split/train", 'rb') as f:
        train = pickle.load(f)
    with open(split_dir + "/split/test", 'rb') as f:
        val = pickle.load(f)


    unlabeled, source, target = pre.XML2arrayRAW("data/" + src + "/" + src + "UN.txt",
                                                 "data/" + dest + "/" + dest + "UN.txt")

    dest_test, source, target = XML2arrayRAW("data/" + dest + "/negative.parsed",
                                             "data/" + dest + "/positive.parsed")
    unlabeled = getClearList(unlabeled)
    train = getClearList(train)


    tok = Tokenizer(nb_words=topWords, split=" ")
    tok.fit_on_texts(train + unlabeled)

    X_train = tok.texts_to_sequences(train)
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_len)
    train_count = 800
    Y_train = [0] * train_count + [1] * train_count
    val = getClearList(val)
    X_val = tok.texts_to_sequences(val)
    X_val = sequence.pad_sequences(X_val, maxlen=max_review_len)
    val_count = 200
    Y_val =  [0] * val_count + [1] * val_count
    dest_test = getClearList(dest_test)
    X_test = tok.texts_to_sequences(dest_test)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_len)
    test_count = 1000
    Y_test = [0]*test_count+[1]*test_count
    #loading the PBLM model without the softmax layer
    modelT = Sequential()
    for i in range(len(model.layers)-1):
        modelT.add(model.layers[i])
        modelT.layers[i].trainable = False
        modelT.layers[i].mask_zero = False



    modelT.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print "\n\n\nmodel embd masking = ",modelT.layers[0].mask_zero
    print "here come the freeze"
    print modelT.summary()

    filters = 250
    kernel_size = 3



    embd = Sequential()
    embedding_vecor_length = embedding_vecor_length_rep
    embd.add(
        Embedding(topWords, embedding_vecor_length, input_length=max_review_len, init='glorot_uniform', mask_zero=True))
    print(embd.summary())

    LSTMlayer = LSTM(hidden_units_num,name='sentLSTM')
    sent_model = Sequential()

    #connecting the PBLM to the LSTM
    sent_model.add(modelT)
    train_data = X_train
    val_data = X_val
    test_data = X_test
    sent_model.add(LSTMlayer)
    sent_model.add(Dense(1, activation='sigmoid'))
    sent_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print sent_model.layers
    print(sent_model.summary())
    model_str = src + "_to_" + dest + "/sent_models_lstm/model_" + str(pivot_num) + "_" +str(
        hidden_units_num) + "_" +str(embedding_vecor_length)+"_" + str(hidden_units_num_rep)+".model"
    filename = model_str
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    #saves the best model
    modelCheckpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True,
                                      save_weights_only=False, mode='min', period=1)

    #stops as soon as the validation loss stops decreasing
    earlyStopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')
    sent_model.fit(train_data, Y_train, validation_data=(val_data, Y_val), nb_epoch=10, batch_size=16,callbacks=[earlyStopping,modelCheckpoint])
    print(sent_model.summary())
    print sent_model.get_config()
    sent_model = load_model(filename)
    val_score, val_acc =sent_model.evaluate(val_data, Y_val, batch_size=16)


    print('val loss:', val_score)
    print('val accuracy:', val_acc)

    test_score, test_acc = sent_model.evaluate(test_data, Y_test, batch_size=16)

    print('Test loss:', test_score)
    print('Test accuracy:', test_acc)

    score_path = src+"_to_"+dest+"/results/lstm/results.txt"
    sentence = "pivots = " + str(pivot_num) + " HU rep " + str(
        hidden_units_num_rep) + " word rep size " + str(embedding_vecor_length_rep) + " sent HU "+ str(hidden_units_num) +  " the val acc " + str(val_acc) + " test acc "+str(test_acc)

    if not os.path.exists(os.path.dirname(score_path)):
        os.makedirs(os.path.dirname(score_path))

    with open(score_path , "a") as myfile:
        myfile.write(sentence+"\n")
