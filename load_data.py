from functions_modules import *
from keras_transformer1 import *
from keras.datasets import mnist
import numpy as np
import pandas as pd
from keras import backend as K
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
import nltk
import re
import sys
from sklearn.decomposition import PCA

def load_transformer_data(num_classes, del_classes_training, del_classes_testing, removeClasses, dataset):
    input_shape = 1,1
    if (dataset == "CCAT-50"):
        path = "./text_data/CCAT-50_Train-All Data.csv"
    elif (dataset == "Amazon"):
        path = "./text_data/AmazonReviews100.csv"
    data = pd.read_csv(path, usecols=["text", "label"], delimiter=',')

    x_train = data["text"].fillna(' ').values.tolist()
    y_train = data["label"].fillna(' ').values.tolist()
    x_train = np.array(x_train + x_train)
    y_train = np.array(y_train + y_train)

    #SPECIFIC
    y_test = y_train
    x_test = x_train
    #x_train = x_train[:20]
    #y_train = y_train[:20]

    # Obtaining new Training
    x = []
    if removeClasses is True:
        for i in del_classes_training:
            y = np.where(y_train == i)
            for j in y[0]:
                x.append(j)
        y = np.where(y_train >= num_classes)
        for j in y[0]:
            x.append(j)
        x.sort(reverse=True)
        x_trainNew = np.delete(x_train, x, 0)
        y_trainNew = np.delete(y_train, x, 0)
    # Obtaining new Testing
    x = []
    if removeClasses is True:
        for i in del_classes_testing:
            y = np.where(y_test == i)
            for j in y[0]:
                x.append(j)
        y = np.where(y_train >= num_classes)
        for j in y[0]:
            x.append(j)
        x.sort(reverse=True)
        x_testNew = np.delete(x_test, x, 0)
        y_testNew = np.delete(y_test, x, 0)

        # Known Unknown Data
        x_knownUnknown = []
        y_knownUnknown = []
        vals = list(set(y_trainNew))

        length = list(y_trainNew).count(vals[-1]) / 1
        for i in vals:
            j = 0;
            temp = 0;
            while temp < length:
                if y_trainNew[j] == i:
                    x_knownUnknown.append(x_trainNew[j])
                    y_knownUnknown.append(y_trainNew[j])
                    temp += 1
                j += 1
        x_knownUnknown = np.array(x_knownUnknown)
        y_knownUnknown = np.array(y_knownUnknown)

    #Data Tokenization
    x_TrainTokens, y_TrainTokens, x_TestTokens, y_TestTokens, x_knownUnknownT, y_knownUnknownT = [],[],[],[],[],[]
    x_All, y_All = [], []
    for i in x_trainNew:
        x_TrainTokens.append(text_as_tokens(str(i)))
    for i in y_trainNew:
        y_TrainTokens.append(text_as_tokens(str(i)))
    for i in x_testNew:
        x_TestTokens.append(text_as_tokens(str(i)))
    for i in y_testNew:
        y_TestTokens.append(text_as_tokens(str(i)))
    for i in x_knownUnknown:
        x_knownUnknownT.append(text_as_tokens(str(i)))
    for i in y_knownUnknown:
        y_knownUnknownT.append(text_as_tokens(str(i)))
    for i in x_test:
        x_All.append(text_as_tokens(str(i)))
    for i in y_test:
        y_All.append(text_as_tokens(str(i)))

    return x_TrainTokens, x_TestTokens, y_TrainTokens, y_TestTokens, x_knownUnknownT, y_knownUnknownT, x_All, y_All, input_shape

def transformerEncoderStep(source_tokens, target_tokens):
    source_token_dict = build_token_dict(source_tokens)
    target_token_dict = build_token_dict(target_tokens)
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

    # Add special tokens
    encode_tokens = [['<START>'] + tokens + ['<END>']
                     for tokens in source_tokens]
    decode_tokens = [['<START>'] + tokens + ['<END>']
                     for tokens in target_tokens]
    output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]

    # Padding
    source_max_len = max(map(len, encode_tokens))
    target_max_len = max(map(len, decode_tokens))

    encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
    decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
    output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]

    encode_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encode_tokens]
    decode_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decode_tokens]
    decode_output = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]

    return encode_input, decode_input, decode_output, source_token_dict, target_token_dict

def load_Normal_data(num_classes, del_classes_training, del_classes_testing, removeClasses, dataset):
    img_rows, img_cols = 30,300
    if(dataset == "CCAT-50"):
        path = "./text_data/CCAT-50_Train-All Data.csv"
    elif(dataset == "Amazon"):
        path = "./text_data/AmazonReviews100.csv"
    data = pd.read_csv(path, usecols=["text", "label"], delimiter=',')

    x_train = data["text"].fillna(' ').values.tolist()
    y_train = data["label"].fillna(' ').values.tolist()
    x_train = np.array(x_train + x_train)
    y_train = np.array(y_train + y_train)

    #SPECIFIC
    y_test = y_train
    x_test = x_train

    # Obtaining new Training
    x = []
    if removeClasses is True:
        for i in del_classes_training:
            y = np.where(y_train == i)
            for j in y[0]:
                x.append(j)
        y = np.where(y_train >= num_classes)
        for j in y[0]:
            x.append(j)
        x.sort(reverse=True)
        x_trainNew = np.delete(x_train, x, 0)
        y_trainNew = np.delete(y_train, x, 0)
    # Obtaining new Testing
    x = []
    if removeClasses is True:
        for i in del_classes_testing:
            y = np.where(y_test == i)
            for j in y[0]:
                x.append(j)
        y = np.where(y_train >= num_classes)
        for j in y[0]:
            x.append(j)
        x.sort(reverse=True)
        x_testNew = np.delete(x_test, x, 0)
        y_testNew = np.delete(y_test, x, 0)

    print(set(y_trainNew))

    #Word Embeddings, and Preprocess Text
    #Max Doc Length = 300
    maxLen = img_rows
    embedding_size = 300
    new_model = False #change to either create new word2vec or load word2vec
    google_vectors = True
    print("Loading Word Vectors")
    if google_vectors == True:
        #wordModel = Word2Vec.load_word2vec_format('./generated_models/GoogleNews-vectors-negative300.bin', binary=True)
        wordModel = word2vec.KeyedVectors.load_word2vec_format("./generated_models/GoogleNews-vectors-negative300.bin", binary=True)
    elif new_model == True and google_vectors == False:
        for i in (range(len(x_trainNew))):
            x_trainNew[i] = x_trainNew[i].lower()
            x_trainNew[i] = re.sub('[^a-zA-Z]', ' ', x_trainNew[i])
        #Creating Word2Vec Model
        all_words = [nltk.word_tokenize(sent) for sent in x_trainNew]
        wordModel = Word2Vec(all_words, min_count=1, size=embedding_size)
        wordModel.save("./generated_models/ccat50_word2vec.model")
    else:
        wordModel = Word2Vec.load("./generated_models/ccat50_word2vec.model")

    print("Finished Loading Word Vectors")
    #Generating Final Word Embedding Data
    x_trainNewEmbed = []
    removedWords = []
    y = []
    for doc in x_trainNew:
        y.clear()
        x = doc.split()
        if len(x) > maxLen:
            x = x[:maxLen]
        for word in x:
            if word in wordModel.wv.vocab:
                z = list(wordModel.wv[word])
                y.append(z)
            else:
                removedWords.append(word)
        while len(y) < maxLen:
            y.append(list([0]*embedding_size))
        x_trainNewEmbed.append(y.copy())
        y.clear()
    x_trainNew = x_trainNewEmbed

    x_testNewEmbed = []
    y = []
    for doc in x_testNew:
        y.clear()
        x = doc.split()
        if len(x) > maxLen:
            x = x[:maxLen]
        for word in x:
            if word in wordModel.wv.vocab:
                z = list(wordModel.wv[word])
                y.append(z)
            else:
                removedWords.append(word)
        while len(y) < maxLen:
            y.append(list([0]*embedding_size))
        x_testNewEmbed.append(y.copy())
        y.clear()
    x_testNew = x_testNewEmbed

    x_trainNew = np.array(x_trainNew)
    x_testNew = np.array(x_testNew)
    x_trainNew = x_trainNew.astype('float32')
    x_testNew = x_testNew.astype('float32')

    #x_trainNew /= 255

    if K.image_data_format() == 'channels_first':
        x_trainNew = x_trainNew.reshape(len(x_trainNew), 1, img_rows, img_cols)
        x_testNew = x_testNew.reshape(len(x_testNew), 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_trainNew = x_trainNew.reshape(len(x_trainNew), img_rows, img_cols, 1)
        x_testNew = x_testNew.reshape(len(x_testNew), img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    '''
    #validation Data
    x_validate = []
    y_validate = []
    vals = list(set(y_testNew))

    length = list(y_testNew).count(vals[-1]) / 5
    for i in vals:
        j = 0; temp = 0;
        while temp < length: #int(len(x_testNew)/10)
            if y_testNew[j] == i:
                x_validate.append(x_testNew[j])
                y_validate.append(y_testNew[j])
                temp += 1
            j += 1
    x_validate = np.array(x_validate)
    y_validate = np.array(y_validate)
    '''
    #Known Unknown Data
    x_knownUnknown = []
    y_knownUnknown = []
    vals = list(set(y_trainNew))

    length = list(y_trainNew).count(vals[-1]) / 2
    for i in vals:
        j = 0; temp = 0;
        while temp < length:
            if y_trainNew[j] == i:
                x_knownUnknown.append(x_trainNew[j])
                y_knownUnknown.append(y_trainNew[j])
                temp += 1
            j += 1
    x_knownUnknown = np.array(x_knownUnknown)
    y_knownUnknown = np.array(y_knownUnknown)

    return x_trainNew, x_testNew, y_trainNew, y_testNew, x_knownUnknown, y_knownUnknown, input_shape

def load_MNIST_data(num_classes, del_classes_training, del_classes_testing, removeClasses):
    # input image dimensions
    #img_rows, img_cols = 28, 28
    img_rows, img_cols = 32, 32

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #SPECIFIC
    y_test = y_train
    x_test = x_train

    # delete specific classes from testing data
    print("For Incremental Learning, use digits (0-4) for training, initially have (0-5) for testing")
    # Obtaining new Training
    x = []
    if removeClasses is True:
        for i in del_classes_training:
            y = np.where(y_train == i)
            for j in y[0]:
                x.append(j)
        x.sort(reverse=True)
        x_trainNew = np.delete(x_train, x, 0)
        y_trainNew = np.delete(y_train, x, 0)
    # Obtaining new Testing
    x = []
    if removeClasses is True:
        for i in del_classes_testing:
            y = np.where(y_test == i)
            for j in y[0]:
                x.append(j)
        x.sort(reverse=True)
        x_testNew = np.delete(x_test, x, 0)
        y_testNew = np.delete(y_test, x, 0)

    ##RESHAPING to 32x32
    x_testNew = x_testNew.tolist()
    for i in range(len(x_testNew)):
        for j in range(len(x_testNew[i])):
            zeros_4 = [0] * 4
            x_testNew[i][j] = x_testNew[i][j] + zeros_4

        zeros_32 = [[0] * 32]
        x_testNew[i] = x_testNew[i] + zeros_32
        x_testNew[i] = x_testNew[i] + zeros_32
        x_testNew[i] = x_testNew[i] + zeros_32
        x_testNew[i] = x_testNew[i] + zeros_32
    x_testNew = np.array(x_testNew)
    x_trainNew = x_trainNew.tolist()
    for i in range(len(x_trainNew)):
        for j in range(len(x_trainNew[i])):
            zeros_4 = [0] * 4
            x_trainNew[i][j] = x_trainNew[i][j] + zeros_4

        zeros_32 = [[0] * 32]
        x_trainNew[i] = x_trainNew[i] + zeros_32
        x_trainNew[i] = x_trainNew[i] + zeros_32
        x_trainNew[i] = x_trainNew[i] + zeros_32
        x_trainNew[i] = x_trainNew[i] + zeros_32
    x_trainNew = np.array(x_trainNew)
    ####End of Reshaping

    if K.image_data_format() == 'channels_first':
        x_trainNew = x_trainNew.reshape(x_trainNew.shape[0], 1, img_rows, img_cols)
        x_testNew = x_testNew.reshape(x_testNew.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_trainNew = x_trainNew.reshape(x_trainNew.shape[0], img_rows, img_cols, 1)
        x_testNew = x_testNew.reshape(x_testNew.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_trainNew = x_trainNew.astype('float32')
    x_testNew = x_testNew.astype('float32')
    x_trainNew /= 255
    x_testNew /= 255

    x_validate = []
    y_validate = []
    vals = set(y_testNew)
    for i in vals:
        j = 0; temp = 0;
        while temp < int(len(x_testNew)/100):
            if y_testNew[j] == i:
                x_validate.append(x_testNew[j])
                y_validate.append(y_testNew[j])
                temp += 1
            j += 1
    x_validate = np.array(x_validate)
    y_validate = np.array(y_validate)

    x_knownUnknown = []
    y_knownUnknown = []
    vals = set(y_trainNew)
    for i in vals:
        j = 0; temp = 0;
        while temp < int(len(x_trainNew)/50):
            if y_trainNew[j] == i:
                x_knownUnknown.append(x_trainNew[j])
                y_knownUnknown.append(y_trainNew[j])
                temp += 1
            j += 1
    x_knownUnknown = np.array(x_knownUnknown)
    y_knownUnknown = np.array(y_knownUnknown)

    return x_trainNew, x_testNew, y_trainNew, y_testNew, x_validate, y_validate, x_knownUnknown, y_knownUnknown, input_shape