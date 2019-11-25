import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

runs = 5
batch_size = 10
epochs = 10
model = "Original"
data_set = "Amazon" #CCAT-50, Amazon

def clear_results_file():
    with open("Results.txt", "w"):
        pass
clear_results_file()

for i in range(runs):
    os.system("python initial_test.py "+str(i)+" "+str(batch_size)+" "+str(epochs)+" "+str(model)+" "+str(data_set))
    #sys.exit()

