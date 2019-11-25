import keras
import tensorflow as tf
from keras.models import load_model
from keras.models import model_from_json
from sklearn.utils import shuffle
from sklearn.cluster import KMeans, DBSCAN, Birch, SpectralClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os
from functions_modules import *
from keras_transformer1 import *
from load_data import *
from keras_transformer import get_custom_objects
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

new_load = "new"
i = int(sys.argv[1])
batch_size1 = int(sys.argv[2])
epochs = int(sys.argv[3])
m = str(sys.argv[4])
data_set = str(sys.argv[5])
num_classes = 10
removeClasses = True
del_classes_train = [5,6,7,8,9] # List of removed classes
del_class1 = [6,7,8,9]
del_class2 = [7,8,9]
del_class3 = [8,9]
del_class4 = [9]
del_class5 = []
del_classList = [del_classes_train,del_class1,del_class2,del_class3,del_class4,del_class5]

sequentialModel = False
if m == "Original":
    print("Original Model")
    new_Model = new_model_original
    previous_node_count = 128 #ResNet-512, Original-128, LeNet-84
elif m == "LeNet":
    print("LeNet-5 Model")
    new_Model = LeNet_5
    previous_node_count = 84
elif m == "ResNet":
    print("ResNet-18 Model")
    new_Model = ResNet_18
    previous_node_count = 512
elif m == "Transformer":
    print("Transformer Model")
    new_Model = Transformer
    previous_node_count = 32####

def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40)
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    if (m != "Transformer"):
        if (i==0):
            model = main_step(del_classList[i], del_classList[i+1], True, None)
            model.save("./generated_models/temp_saves/model.h5")
        else:
            model = load_model("./generated_models/temp_saves/model.h5")
            model = main_step(del_classList[i], del_classList[i+1], False, model)
            model.save("./generated_models/temp_saves/model.h5")
    elif (m == "Transformer"):
        if (i==0):
            model = main_step(del_classList[i], del_classList[i + 1], True, None)
            model.save_weights('./generated_models/temp_saves/model_weights.h5')
            with open('./generated_models/temp_saves/model.json', 'w') as fh:
                fh.write(model.to_json())
        else:
            with open('./generated_models/temp_saves/model.json', 'r') as fh:
                model = model_from_json(fh.read(), get_custom_objects())
            model.load_weights('./generated_models/temp_saves/model_weights.h5')
            model = main_step(del_classList[i], del_classList[i + 1], False, model)
            model.save_weights('./generated_models/temp_saves/model_weights.h5')
            with open('./generated_models/temp_saves/model.json', 'w') as fh:
                fh.write(model.to_json())

def main_step(del_classes_training, del_classes_testing, start, model):
    DATA_MULTIPLIER = 1  # 1024
    batch_size = batch_size1
    new_class_val = list(set(del_classes_training) - set(del_classes_testing))[0]

    num_classes_train = num_classes - len(del_classes_training)
    num_classes_test = num_classes - len(del_classes_testing)
    x_validate, y_validate = [], []

    if m == "Transformer":
        x_trainNew, x_testNew, y_trainNew, y_testNew, x_knownUnknown, y_knownUnknown, x_All, y_All, input_shape = load_transformer_data(
            num_classes, del_classes_training, del_classes_testing, removeClasses, data_set)
    else:
        x_trainNew, x_testNew, y_trainNew, y_testNew, x_knownUnknown, y_knownUnknown, input_shape = load_Normal_data(
            num_classes, del_classes_training, del_classes_testing, removeClasses, data_set)

    #x_trainNew, x_testNew, y_trainNew, y_testNew, x_validate, y_validate, x_knownUnknown, y_knownUnknown, input_shape = load_MNIST_data(
     #   num_classes, del_classes_training, del_classes_testing, removeClasses)

    # convert class vectors to binary class matrices
    if m != "Transformer":
        y_trainNew_Cat = keras.utils.to_categorical(y_trainNew, num_classes_train)
        y_testNew_Cat = keras.utils.to_categorical(y_testNew, num_classes_test)
        #y_validate_Cat = keras.utils.to_categorical(y_validate, num_classes_test)
        #x_validate, y_validate = shuffle(x_validate, y_validate, random_state=0)  # Random Shuffle All Validate Data

    if start == True and m != "Transformer":
        model = new_Model(num_classes_train, num_classes_test, input_shape, False)
        model.fit(x_trainNew, y_trainNew_Cat,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1) #validation_split=0.1
    elif start == True and m == "Transformer":
        encode_input, decode_input, decode_output, source_token_dict, target_token_dict = transformerEncoderStep(x_trainNew, y_trainNew)
        encode_inputTest, decode_inputTest, decode_outputTest, source_token_dictTest, target_token_dictTest = transformerEncoderStep(x_testNew, y_testNew)
        encode_inputAll, decode_inputAll, decode_outputAll, source_token_dictAll, target_token_dictAll = transformerEncoderStep(x_All, y_All)
        model = new_Model(num_classes_train, num_classes_test, input_shape, False, source_token_dictAll, target_token_dictAll)
        model.fit(
            x=[np.array(encode_input * DATA_MULTIPLIER), np.array(decode_input * DATA_MULTIPLIER)],
            y=np.array(decode_output * DATA_MULTIPLIER),
            epochs=epochs,
            batch_size=batch_size#, #64
            #validation_split=0.1
        )

    if m == "Transformer":######Below
        if start != True:
            encode_inputTest, decode_inputTest, decode_outputTest, source_token_dictTest, target_token_dictTest = transformerEncoderStep(x_testNew, y_testNew)
            encode_inputAll, decode_inputAll, decode_outputAll, source_token_dictAll, target_token_dictAll = transformerEncoderStep(x_All, y_All)
        modelNew = add_unknown_node(num_classes_train, num_classes_test, model, input_shape, m, source_token_dictAll, target_token_dictAll)
    else:
        modelNew = add_unknown_node(num_classes_train, num_classes_test, model, input_shape, m, 0, 0)

    #Testing
    if m == "Transformer":
        valT, weight_class_val = transformerPredict(encode_inputTest, target_token_dictTest, modelNew)
        weight_class_val2 = weight_class_val
        val = valT
        val2 = valT
    else:
        weight_class_val = model.predict(x_testNew)
        weight_class_val2 = modelNew.predict(x_testNew)
        val = weight_class_val.argmax(axis=-1)
        val2 = weight_class_val2.argmax(axis=-1)

    # initial test of finding unknown data,
    # basic implemenation that says if class element is not 2x greater that next weight, make unknown

    x_testNewU = []
    y_testNewU = []
    test = []
    weight_class_val2 = boost_weights(weight_class_val2, y_testNew, m)
    count = 5
    for i in range(len(weight_class_val2)):
        i_new = weight_class_val2[i]

        if (i_new[-1] < i_new[-2] * 10):  #i_new[-1] < i_new[-2] * 500
            if m != "Transformer":
                for j in range(count): #Number of times to add unknown identified for further training
                    x_testNewU.append(x_testNew[i])
                    y_testNewU.append(new_class_val)  # y_testNew[i] #new_class_val
            else:
                #for j in range(count):
                    x_testNewU.append(x_testNew[i])
                    y_testNewU.append([str(new_class_val)])
                    test.append(y_testNew[i])
                    #y_testNewU.append(y_testNew[i])

    if m != "Transformer":
        print("New Class Val: ", str(new_class_val))  ########
        for i in range(len(set(val2)) + 1):
            print(str(i) + ": " + str(list(y_testNewU).count(i) / count))
        print("Number of Classes: " + str(len(set(val2))))
        print("Len Known Unknown per Class: " + str(len(y_knownUnknown) / len(set(val2))))
    else:
        print("Number of unknown")
        print(len(y_testNewU))
        print("Number of unknowns obtained")
        print(len(y_testNewU)/count)
        print(y_testNewU)
        print(test)

    if m != "Transformer":
        x_testNewU = np.array(x_testNewU)
        y_testNewU = np.array(y_testNewU)
        x_testNewU = np.concatenate((x_testNewU, x_knownUnknown), axis=0)  # Adding KnownUnknown Data to mix of Unknown
        y_testNewU = np.concatenate((y_testNewU, y_knownUnknown), axis=0)
        x_testNewU, y_testNewU = shuffle(x_testNewU, y_testNewU, random_state=0)  # Random Shuffle All Unknown Data
        y_testNew_CatU = keras.utils.to_categorical(y_testNewU, num_classes_test)
    elif m == "Transformer":
        x_testNewU = x_knownUnknown + x_testNewU
        y_testNewU = y_knownUnknown + y_testNewU
        print(len(y_testNewU))

        # x_testNewU, y_testNewU = shuffle(x_testNewU, y_testNewU, random_state=0)  # Random Shuffle All Unknown Data

    if m != "Transformer":
        print(num_classes_test)
        print(set(list(y_testNewU)))
        print(set(list(y_knownUnknown)))

    # Further Training
    if m != "Transformer":
        modelNew.fit(x_testNewU, y_testNew_CatU,
                     batch_size=batch_size,
                     epochs=epochs-5, #Mnist original e = 3, new = 8, CCAT-50 original e = 10, new = 5
                     verbose=1)#,
                     #validation_data=(x_validate, y_validate_Cat))
    elif m == "Transformer":
        encode_input, decode_input, decode_output, source_token_dict, target_token_dict = transformerEncoderStep(x_testNewU, y_testNewU)
        modelNew.fit(
            x=[np.array(encode_input * DATA_MULTIPLIER), np.array(decode_input * DATA_MULTIPLIER)],
            y=np.array(decode_output * DATA_MULTIPLIER),
            epochs=epochs,  # +10
            batch_size=batch_size#,  # 64
            #validation_split=0.1
        )

    if m == "Transformer":
        val3, weight_class_val3 = transformerPredict(encode_inputTest, target_token_dictTest, modelNew)
    else:
        weight_class_val3 = modelNew.predict(x_testNew)
        val3 = weight_class_val3.argmax(axis=-1)

    print_results(val, val2, val3, y_testNew, new_class_val, m)
    print("Iteration Complete")
    return modelNew

def boost_weights(input_weights, y_testNew, m):
    #length = len(input_weights[0])
    #mid = int(length / 2)
    input_weights_new = []

    '''
    k_model = KMeans(n_clusters=6)
    k_model.fit(input_weights)
    k_labels = k_model.predict(input_weights)
    db_model = DBSCAN(eps=0.1)#130
    db_model.fit(input_weights)
    db_labels = db_model.labels_
    plot_points = PCA(n_components=2).fit_transform(input_weights)
    x1 = [i[0] for i in plot_points]#; x1 = [i/100 for i in x1]
    y1 = [i[1] for i in plot_points]#; y1 = [i/100 for i in y1]

    colormap = np.array(['red', 'lime', 'blue', 'purple', 'green', 'yellow', 'orange', 'pink'])
    plt.subplot(1, 3, 1)
    plt.scatter(x1, y1, c=colormap[y_testNew], s=30)
    plt.subplot(1, 3, 2)
    plt.scatter(x1, y1, c=colormap[k_labels], s=30)
    plt.subplot(1, 3, 3)
    plt.scatter(x1, y1, c=colormap[db_labels], s=30)
    plt.show()
    sys.exit()
    '''
    if (m != "Transformer"):
        for i in range(len(input_weights)):
            input_weights[i].sort()
            temp = 0
            for j in range(len(input_weights[i])-1):
                temp += input_weights[i][j]
            temp = temp / (len(input_weights[i])-1)
            input_weights_new.append([temp, input_weights[i][-1]])
    else: # Work Here Tomorrow!
        t = 0
        for i in range(len(input_weights)):
            input_weights[i][t].sort()
            temp = 0
            for j in range(len(input_weights[i][t])-1):
                temp += input_weights[i][t][j]
            temp = temp / (len(input_weights[i][t])-1)
            input_weights_new.append([temp, input_weights[i][t][-1]])

    return input_weights_new

def add_unknown_node(num_classes_train, num_classes_test, modelOriginal, input_shape, m, source_token_dict, target_token_dict):
    new_weight = 0.00000001
    weights = [] #all weights except Softmax laye
    for i in range(len(modelOriginal.layers)-1): #Get weights from orignal model except Softmax layer
        weights.append(modelOriginal.layers[i].get_weights())

    if m == "Transformer":
        model = new_Model(num_classes_train, num_classes_test, input_shape, True, source_token_dict, target_token_dict)
    else:
        model = new_Model(num_classes_train, num_classes_test, input_shape, True) #Generate new updated model

    # Generate new 'Unknown' Node to last layer
    append_arry = [new_weight]*previous_node_count
    new_softLayer_default = modelOriginal.layers[-1].get_weights()[0]
    new_softLayer_default = np.column_stack((new_softLayer_default, append_arry))

    softLayer = modelOriginal.layers[-1].get_weights()
    softLayer_firstHalf = new_softLayer_default
    softLayer_secondHalf = softLayer[1]
    softLayer_secondHalf = np.append(softLayer_secondHalf, new_weight)  # Adding new weight
    new_softLayer = [softLayer_firstHalf, softLayer_secondHalf]

    for i in range(len(modelOriginal.layers)-1):
        try:
            model.layers[i].set_weights(weights[i])
        except:
            print(i)

    model.layers[-1].set_weights(new_softLayer)

    return model

def print_results(val, val2, val3, y_testNew, new_class_val, m):
    # Printing Results
    results = [0, 0, 0]
    for i in range(len(y_testNew)):  # Regular Acc
        if m != "Transformer":
            if int(val[i]) == int(y_testNew[i]):
                results[0] += 1
            if int(val2[i]) == int(y_testNew[i]):
                results[1] += 1
            if int(val3[i]) == int(y_testNew[i]):
                results[2] += 1
        elif m == "Transformer":
            if int(val[i][0]) == int(y_testNew[i][0]):
                results[0] += 1
            if int(val2[i][0]) == int(y_testNew[i][0]):
                results[1] += 1
            if int(val3[i][0]) == int(y_testNew[i][0]):
                results[2] += 1
    with open("Results.txt", 'a') as file:
        file.write("Adding Class: %d\n" %new_class_val)
        x = results[0] / len(y_testNew)
        print("Training Accuracy Original: ", str(x)); #print(set(val));
        file.write("Training Accuracy Original: %f\n" %x)

        x = results[1] / len(y_testNew)
        print("Training Accuracy New: ", str(x)); #print(set(val2));
        file.write("Training Accuracy New: %f\n" %x)

        x = results[2] / len(y_testNew)
        print("Training Accuracy Retrained: ", str(x)); #print(set(val3));
        file.write("Training Accuracy Retrained: %f\n" %x)

    print(val)
    print(val3)
    print(y_testNew)

def _get_max_suffix_repeat_times(tokens, max_len):
    detect_len = min(max_len, len(tokens))
    next = [-1] * detect_len
    k = -1
    for i in range(1, detect_len):
        while k >= 0 and tokens[len(tokens) - i - 1] != tokens[len(tokens) - k - 2]:
            k = next[k]
        if tokens[len(tokens) - i - 1] == tokens[len(tokens) - k - 2]:
            k += 1
        next[i] = k
    max_repeat = 1
    for i in range(2, detect_len):
        if next[i] >= 0 and (i + 1) % (i - next[i]) == 0:
            max_repeat = max(max_repeat, (i + 1) // (i - next[i]))
    return max_repeat

def transformerPredict(encode_input, target_token_dict, model):
    tokens = encode_input  ####Testing Values
    start_token = target_token_dict['<START>']
    pad_token = target_token_dict['<PAD>']
    end_token = target_token_dict['<END>']

    is_single = not isinstance(tokens[0], list)
    if is_single:
        tokens = [tokens]
    batch_size = len(tokens)
    decoder_inputs = [[start_token] for _ in range(batch_size)]
    outputs = [None for _ in range(batch_size)]
    output_len = 1
    start = True
    while len(list(filter(lambda x: x is None, outputs))) > 0:
        output_len += 1
        batch_inputs, batch_outputs = [], []
        max_input_len = 0
        index_map = {}
        for i in range(batch_size):
            if outputs[i] is None:
                index_map[len(batch_inputs)] = i
                batch_inputs.append(tokens[i][:])
                batch_outputs.append(decoder_inputs[i])
                max_input_len = max(max_input_len, len(tokens[i]))
        for i in range(len(batch_inputs)):
            batch_inputs[i] += [pad_token] * (max_input_len - len(batch_inputs[i]))
        if start == True:
            weight_class_val = model.predict([np.array(batch_inputs), np.array(batch_outputs)])
            start = False
        predicts = model.predict([np.array(batch_inputs), np.array(batch_outputs)])

        top_k = 1
        temperature = 1.0
        max_len = 10000
        max_repeat = 10
        max_repeat_block = 10
        for i in range(len(predicts)):  ###########
            if top_k == 1:
                last_token = predicts[i][-1].argmax(axis=-1)
            else:
                probs = [(prob, j) for j, prob in enumerate(predicts[i][-1])]
                probs.sort(reverse=True)
                probs = probs[:top_k]
                indices, probs = list(map(lambda x: x[1], probs)), list(map(lambda x: x[0], probs))
                probs = np.array(probs) / temperature
                probs = probs - np.max(probs)
                probs = np.exp(probs)
                probs = probs / np.sum(probs)
                last_token = np.random.choice(indices, p=probs)
            decoder_inputs[index_map[i]].append(last_token)
            if last_token == end_token or \
                    (max_len is not None and output_len >= max_len) or \
                    _get_max_suffix_repeat_times(decoder_inputs, max_repeat * max_repeat_block) >= max_repeat:
                outputs[index_map[i]] = decoder_inputs[index_map[i]]
    if is_single:
        outputs = outputs[0]

    valT = []
    for i in outputs:
        valT.append([str(int(i[1])-3)])

    return valT, weight_class_val

if __name__ == "__main__":
    main()