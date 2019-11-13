import keras
import tensorflow as tf
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.cluster import KMeans, DBSCAN, Birch, SpectralClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import sys
from functions_modules import *
from keras_transformer1 import *
from load_data import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

new_load = "new"
i = int(sys.argv[1])
batch_size1 = int(sys.argv[2])
epochs = int(sys.argv[3])
m = str(sys.argv[4])
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
    new_Model = new_model_original
    previous_node_count = 128 #ResNet-512, Original-128, LeNet-84
elif m == "LeNet":
    new_Model = LeNet_5
    previous_node_count = 84
elif m == "RedNet":
    new_Model = ResNet_18
    previous_node_count = 512
elif m == "Transformer":
    new_Model = Transformer
    previous_node_count = 32####

def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    if (i==0):
        model = main_step(del_classList[i], del_classList[i+1], True, None)
        model.save("./generated_models/temp_saves/model.h5")
    else:
        model = load_model("./generated_models/temp_saves/model.h5")
        model = main_step(del_classList[i], del_classList[i+1], False, model)
        model.save("./generated_models/temp_saves/model.h5")


def main_step(del_classes_training, del_classes_testing, start, model):
    batch_size = batch_size1
    new_class_val = list(set(del_classes_training) - set(del_classes_testing))[0]

    num_classes_train = num_classes - len(del_classes_training)
    num_classes_test = num_classes - len(del_classes_testing)
    x_validate, y_validate = [], []

    if m == "Transformer":
        x_trainNew, x_testNew, y_trainNew, y_testNew, x_knownUnknown, y_knownUnknown, input_shape = load_CCAT50_data_transformer(
            num_classes, del_classes_training, del_classes_testing, removeClasses)
    else:
        x_trainNew, x_testNew, y_trainNew, y_testNew, x_validate, y_validate, x_knownUnknown, y_knownUnknown, input_shape = load_CCAT50_data(
            num_classes, del_classes_training, del_classes_testing, removeClasses)

    #x_trainNew, x_testNew, y_trainNew, y_testNew, x_validate, y_validate, x_knownUnknown, y_knownUnknown, input_shape = load_MNIST_data(
     #   num_classes, del_classes_training, del_classes_testing, removeClasses)

    # convert class vectors to binary class matrices
    if m != "Transformer":
        y_trainNew_Cat = keras.utils.to_categorical(y_trainNew, num_classes_train)
        y_testNew_Cat = keras.utils.to_categorical(y_testNew, num_classes_test)
        y_validate_Cat = keras.utils.to_categorical(y_validate, num_classes_test)
        x_validate, y_validate = shuffle(x_validate, y_validate, random_state=0)  # Random Shuffle All Validate Data

    if start == True and m != "Transformer":
        model = new_Model(num_classes_train, num_classes_test, input_shape, False)
        model.fit(x_trainNew, y_trainNew_Cat,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1) #validation_split=0.1
    elif start == True and m == "Transformer":
        print(epochs)
        print(batch_size)
        encode_input, decode_input, decode_output, source_token_dict, target_token_dict = transformerEncoderStep(x_trainNew, y_trainNew)
        encode_inputTest, decode_inputTest, decode_outputTest, source_token_dictTest, target_token_dictTest = transformerEncoderStep(x_testNew, y_testNew)
        model = new_Model(num_classes_train, num_classes_test, input_shape, False, source_token_dict, target_token_dict)
        DATA_MULTIPLIER = 1#1024
        model.fit(
            x=[np.array(encode_input * DATA_MULTIPLIER), np.array(decode_input * DATA_MULTIPLIER)],
            y=np.array(decode_output * DATA_MULTIPLIER),
            epochs=epochs-9, #+10
            batch_size=batch_size-1, #64
            validation_split=0.1
        )

    if m == "Transformer":######Below
        #modelNew = new_Model(num_classes_train, num_classes_test, input_shape, True, source_token_dictTest, target_token_dictTest)
        modelNew = add_unknown_node(num_classes_train, num_classes_test, model, input_shape, m, source_token_dictTest, target_token_dictTest)
    else:
        modelNew = add_unknown_node(num_classes_train, num_classes_test, model, input_shape, m, 0, 0)

    if m == "Transformer":
        tokens = encode_inputTest
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
            weight_class_val = model.predict([np.array(batch_inputs), np.array(batch_outputs)])
            weight_class_val2 = modelNew.predict([np.array(batch_inputs), np.array(batch_outputs)])
            weight_class_val = weight_class_val.reshape(len(weight_class_val), len(weight_class_val[0][0]))
            weight_class_val2 = weight_class_val2.reshape(len(weight_class_val2), len(weight_class_val2[0][0]))
            temp1, temp2 = [], []
            for i in weight_class_val:
                temp1.append(i[3:len(i)])
            for i in weight_class_val2:
                temp2.append(i[3:len(i)])
            temp1 = np.array(temp1); temp2 = np.array(temp2)
            val = temp1.argmax(axis=-1)
            val2 = temp2.argmax(axis=-1)
            break
    else:
        weight_class_val = model.predict(x_testNew)
        weight_class_val2 = modelNew.predict(x_testNew)
        val = weight_class_val.argmax(axis=-1)
        val2 = weight_class_val2.argmax(axis=-1)

    # initial test of finding unknown data,
    # basic implemenation that says if class element is not 2x greater that next weight, make unknown
    x_testNewU = []
    y_testNewU = []
    weight_class_val2 = boost_weights(weight_class_val2, y_testNew, m)
    ct = 0
    for i in range(len(weight_class_val2)):
        i_new = weight_class_val2[i]

        if (i_new[-1] < i_new[-2] * 10):  # and y_testNew[i] == 5
            if m != "Transformer":
                for j in range(10): #Number of times to add unknown identified for further training
                    x_testNewU.append(x_testNew[i])
                    y_testNewU.append(new_class_val)  # y_testNew[i] #new_class_val
            elif m == "Transformer":
                x_testNewU.append(x_testNew[i])
                y_testNewU.append([str(new_class_val)])

    print("New Class Val: ", str(new_class_val))
    for i in range(len(set(val2))+1):
        print(str(i) + ": " + str(list(y_testNewU).count(i)))
    print("Len Known Unknown: " + str(len(y_knownUnknown)))

    if m != "Transformer":
        x_testNewU = np.array(x_testNewU)
        y_testNewU = np.array(y_testNewU)
        x_testNewU = np.concatenate((x_testNewU, x_knownUnknown), axis=0)  # Adding KnownUnknown Data to mix of Unknown
        y_testNewU = np.concatenate((y_testNewU, y_knownUnknown), axis=0)
        x_testNewU, y_testNewU = shuffle(x_testNewU, y_testNewU, random_state=0)  # Random Shuffle All Unknown Data
        y_testNew_CatU = keras.utils.to_categorical(y_testNewU, num_classes_test)
    elif m == "Transformer":
        x_testNewU = x_testNewU + x_knownUnknown
        y_testNewU = y_testNewU + y_knownUnknown

    if m != "Transformer":
        print(num_classes_test)
        print(set(list(y_testNewU)))
        print(set(list(y_knownUnknown)))
        print(set(y_validate))

    # Further Training
    if m != "Transformer":
        modelNew.fit(x_testNewU, y_testNew_CatU,
                     batch_size=batch_size,
                     epochs=epochs-5, #Mnist original e = 3, new = 8, CCAT-50 original e = 10, new = 5
                     verbose=1)#,
                     #validation_data=(x_validate, y_validate_Cat))
    elif m == "Transformer":
        encode_input, decode_input, decode_output, source_token_dict, target_token_dict = transformerEncoderStep(x_testNewU, y_testNewU)
        DATA_MULTIPLIER = 1  # 1024
        modelNew.fit(
            x=[np.array(encode_input * DATA_MULTIPLIER), np.array(decode_input * DATA_MULTIPLIER)],
            y=np.array(decode_output * DATA_MULTIPLIER),
            epochs=epochs - 9,  # +10
            batch_size=batch_size-1,  # 64
            validation_split=0.1
        )

    if m == "Transformer":
        tokens = encode_input
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
            weight_class_val = modelNew.predict([np.array(batch_inputs), np.array(batch_outputs)])
            weight_class_val = weight_class_val.reshape(len(weight_class_val), len(weight_class_val[0][0]))
            temp1 = []
            for i in weight_class_val:
                temp1.append(i[3:len(i)])
            temp1 = np.array(temp1)
            val3 = temp1.argmax(axis=-1)
            break
    else:
        weight_class_val3 = modelNew.predict(x_testNew)
        val3 = weight_class_val3.argmax(axis=-1)

    print_results(val, val2, val3, y_testNew, new_class_val, m)
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

    for i in range(len(input_weights)):
        input_weights[i].sort()
        temp = 0
        for j in range(len(input_weights[i])-1):
            temp += input_weights[i][j]
        temp = temp / (len(input_weights[i])-1)
        input_weights_new.append([temp, input_weights[i][-1]])

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
        if val[i] == y_testNew[i]:
            results[0] += 1
        if val2[i] == y_testNew[i]:
            results[1] += 1
        if val3[i] == y_testNew[i]:
            results[2] += 1
    #print(val3[:40])
    #print(y_testNew[:40])
    with open("Results_MNIST.txt", 'a') as file:
        file.write("Adding Class: %d\n" %new_class_val)
        x = results[0] / len(y_testNew)
        print("Training Accuracy Original: ", str(x)); print(set(val));
        file.write("Training Accuracy Original: %f\n" %x)
        #file.write(set(val)); file.write("\n");

        x = results[1] / len(y_testNew)
        print("Training Accuracy New: ", str(x)); print(set(val2));
        file.write("Training Accuracy New: %f\n" %x)
        #file.write(set(val2)); file.write("\n");

        x = results[2] / len(y_testNew)
        print("Training Accuracy Retrained: ", str(x)); print(set(val3));
        file.write("Training Accuracy Retrained: %f\n" %x)
        #file.write(set(val3)); file.write("\n");

"""
Objectosphere loss function.
"""
'''
def ring_loss(y_true,y_pred):
    knownsMinimumMag = Input((1,), dtype='float32', name='knownsMinimumMag')
    print(knownsMinimumMag)
    pred = K.sqrt(K.sum(K.square(y_pred),axis=1))
    print(pred)
    error=K.mean(K.square(
        # Loss for Knowns having magnitude greater than knownsMinimumMag
        y_true[:,0]*(K.maximum(knownsMinimumMag-pred,0.))
        # Add two losses
        +
        # Loss for unKnowns having magnitude greater than unknownsMaximumMag
        y_true[:,1]*pred
    ))
    return error
'''


if __name__ == "__main__":
    main()