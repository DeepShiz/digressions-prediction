import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
import keras.backend as K

num_files_in_dir_goodcasts_june = 1440
num_files_in_dir_breakouts_june = 21
num_files_in_dir_breakouts_july = 21
#num_files_in_dir_goodcasts_july = 1200
num_split = 10

data_june = []
output_june = []
predict_test_june = []
breakout_checker = []

path_good_june = str(num_split) + "_normalized_refined_lfc/june/goodcasts/"
#path_good_july = str(num_split) + "_normalized_refined_lfc/july/goodcasts/"
path_breakouts_june = str(num_split) + "_normalized_refined_lfc/june/breakouts/"
path_breakouts_july = str(num_split) + "_normalized_refined/july/breakouts/"

dict_intensity = {1: 16, 2: 15, 3: 14, 4: 13, 5: 12, 6: 11, 7: 10, 8: 9, 9: 8, 10: 7, 11: 6, 12: 5, 13: 4, 14: 3, 15: 2, 16: 1}

intensity_level = []

for i in range(1,num_files_in_dir_goodcasts_june+1):

    if i == 1:

        data_read = pd.read_csv(path_good_june + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_june = data_vals.reshape(1,size[0],size[1])
        output_june.append(0)
        breakout_checker.append(0)

    else:

        data_read = pd.read_csv(path_good_june + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_vals = data_vals.reshape(1,size[0],size[1])
        data_june = np.append(data_june,data_vals,axis=0)
        output_june.append(0)
        breakout_checker.append(0)

for i in range(1,num_files_in_dir_breakouts_june+1):

        data_read = pd.read_csv(path_breakouts_june + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_vals = data_vals.reshape(1,size[0],size[1])
        data_june = np.append(data_june,data_vals,axis=0)
        output_june.append(1)
        breakout_checker += [i]



for i in range(1,num_files_in_dir_breakouts_july+1):

        data_read = pd.read_csv(path_breakouts_july + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_vals = data_vals.reshape(1,size[0],size[1])
        data_june = np.append(data_june,data_vals,axis=0)
        output_june.append(1)
        breakout_checker += [i]

output_june = np.array(output_june)


np.random.seed(99)
np.random.shuffle(data_june)
np.random.seed(99)
np.random.shuffle(output_june)
np.random.seed(99)
np.random.shuffle(breakout_checker)

print(breakout_checker, output_june)

model = load_model("10_tanh_1000epochs_sgd_001_90samps_nomar.h5")

for i in range(data_june.shape[0]):
    k = data_june[i].reshape(1,num_split,36)
    predict_test_june.append(model.predict(k))

print(predict_test_june)

max_val = max(predict_test_june)
min_val = min(predict_test_june)

for i in range(len(predict_test_june)):

    if(predict_test_june[i]<0.5):
        predict_test_june[i] = 0
    elif(predict_test_june[i]>=0.5):
        predict_test_june[i] = 1

count = 0
conf_test = confusion_matrix(output_june, predict_test_june)

for i in range(output_june.shape[0]):
    print([predict_test_june[i],output_june[i]])
    if((output_june[i] == 1) and (predict_test_june[i] != output_june[i])):
        bo_file = breakout_checker[i]
        try:
            intensity_level.append(dict_intensity[bo_file%16])
        except:
            intensity_level.append(dict_intensity[16])
            #continue
    if predict_test_june[i] == output_june[i]:
        count = count+1

print(len(intensity_level), intensity_level)
print(conf_test)
accuracy = 100*count/len(predict_test_june)
print("The accuracy is " + str(accuracy) + "%")
