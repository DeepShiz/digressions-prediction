import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.regularizers import l1
import keras.backend as K


'''num_files_in_dir_goodcasts_may = 480
num_files_in_dir_breakouts_may = 48

num_files_in_dir_breakouts_march = 48
num_files_in_dir_breakouts_april = 32
num_files_in_dir_breakouts_aug = 32

num_split = 10
'''
num_split = 10

num_files_in_dir_goodcasts_march = 1440
num_files_in_dir_goodcasts_april = 1440
num_files_in_dir_goodcasts_may = 1440

num_files_in_dir_breakouts_may = 63
num_files_in_dir_breakouts_march = 63
num_files_in_dir_breakouts_april = 42
#num_files_in_dir_breakouts_aug = 42


predict = []
count = 0

'''path_good_may = "normalized/may/goodcasts/"
path_breakouts_may = "normalized/may/breakouts/"
path_breakouts_july = "normalized/july/breakouts/"
path_breakouts_aug = "normalized/aug/breakouts/"
'''
path_good_march = str(num_split) + "_normalized_refined_lfc/march/goodcasts/"
path_good_april = str(num_split) + "_normalized_refined_lfc/april/goodcasts/"
path_good_may = str(num_split) + "_normalized_refined_lfc/may/goodcasts/"
path_breakouts_march = str(num_split) + "_normalized_refined_lfc/march/breakouts/"
path_breakouts_april = str(num_split) + "_normalized_refined_lfc/april/breakouts/"
path_breakouts_may = str(num_split) + "_normalized_refined_lfc/may/breakouts/"
#path_breakouts_aug = str(num_split) + "_normalized_refined_lfc/aug/breakouts/"

output_train = []
data_input = []
for i in range(1,num_files_in_dir_goodcasts_march+1):
    if i == 1:
        data_read = pd.read_csv(path_good_march + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_input = data_vals.reshape(1,size[0],size[1])
        output_train.append(0)
    else:
        data_read = pd.read_csv(path_good_march + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_vals = data_vals.reshape(1,size[0],size[1])
        data_input = np.append(data_input,data_vals,axis=0)
        output_train.append(0)

for i in range(1,num_files_in_dir_breakouts_march+1):
        data_read = pd.read_csv(path_breakouts_march + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_vals = data_vals.reshape(1,size[0],size[1])
        data_input = np.append(data_input,data_vals,axis=0)
        output_train.append(1)



for i in range(1,num_files_in_dir_goodcasts_april+1):
    if i == 1:
        data_read = pd.read_csv(path_good_april + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_input = np.append(data_input,data_vals.reshape(1,size[0],size[1]),axis=0)
        output_train.append(0)
    else:
        data_read = pd.read_csv(path_good_april + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_vals = data_vals.reshape(1,size[0],size[1])
        data_input = np.append(data_input,data_vals,axis=0)
        output_train.append(0)



for i in range(1,num_files_in_dir_breakouts_april+1):
        data_read = pd.read_csv(path_breakouts_april + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_vals = data_vals.reshape(1,size[0],size[1])
        data_input = np.append(data_input,data_vals,axis=0)
        output_train.append(1)


for i in range(1,num_files_in_dir_goodcasts_may+1):
    if i == 1:
        data_read = pd.read_csv(path_good_may + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_input = np.append(data_input,data_vals.reshape(1,size[0],size[1]),axis=0)
        output_train.append(0)

    else:
        data_read = pd.read_csv(path_good_may + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_vals = data_vals.reshape(1,size[0],size[1])
        data_input = np.append(data_input,data_vals,axis=0)
        output_train.append(0)

for i in range(1,num_files_in_dir_breakouts_may+1):
        data_read = pd.read_csv(path_breakouts_may + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_vals = data_vals.reshape(1,size[0],size[1])
        data_input = np.append(data_input,data_vals,axis=0)
        output_train.append(1)

'''for i in range(1,num_files_in_dir_breakouts_aug+1):

        data_read = pd.read_csv(path_breakouts_aug + str(i) + ".csv",header= None)
        data_vals = data_read.to_numpy()
        size = data_vals.shape
        data_vals = data_vals.reshape(1,size[0],size[1])
        data_input = np.append(data_input,data_vals,axis=0)
        output_train.append(1)'''


# data is our input to the lstm
output_train = np.array(output_train)


np.random.seed(99)
np.random.shuffle(data_input)
np.random.seed(99)
np.random.shuffle(output_train)

print(len(data_input), len(output_train))

print(data_input.shape, output_train.shape)
x_train, x_val, y_train, y_val = train_test_split(data_input, output_train, test_size=0.2, random_state=42)
print(len(x_train), len(y_train), len(x_val), len(y_val))
print(x_train, list(y_train).count(1), y_val)



#Training LSTM model

model = Sequential()
model.add(LSTM(100, input_shape=(num_split, 36)))
model.add(Dense(20, activation="relu"))
#model.add(Activation("relu"))
model.add(Dense(1, activation="sigmoid"))


print(model.summary())

#sgd = optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
sgd = optimizers.SGD(lr=0.005, decay=0, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics = ['accuracy'])

print(K.eval(model.optimizer.lr))

print(x_train.shape)
history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=1000, batch_size=64)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save("final.h5")

print(x_val.shape)
for i in range(x_val.shape[0]):
    k = x_val[i].reshape(1,num_split,36)
    predict.append(float(model.predict(k)))

max_val = max(predict)
min_val = min(predict)

print(predict)

for i in range(len(predict)):

    if(predict[i]<0.5):
        predict[i] = 0
    elif(predict[i]>=0.5):
        predict[i] = 1

conf = confusion_matrix(y_val, predict)
for i in range(y_val.shape[0]):
    print([predict[i],y_val[i]])
    if predict[i] == y_val[i]:
            count = count+1

accuracy = 100*count/len(predict)
print("The accuracy is " + str(accuracy) + "%")

print("\n\nConfusion Matrix\n\n")
print("\t\t0(Pred)\t1(Pred)")
print("\t0(Val)\t" + str(conf[0][0]) + '\t' + str(conf[0][1]))
print("\t1(Val)\t" + str(conf[1][0]) + '\t' + str(conf[1][1]))
