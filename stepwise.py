import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt

data = pd.read_csv("data_breakouts_june.csv",header=None)
data = data.to_numpy()
print(data.shape)

model = load_model("10_tanh_1000epochs_sgd_001_90samps_nomar.h5")

predict = []
time = []
for i in range(10,len(data),1):
    #print(data[i])
    try:
        k = data[i-10:i].reshape(1,10,36)
        predict.append(model.predict(k))
    except:
        continue

for i in range(1, len(predict)+1):
    time += [i]

print(predict)
for i in range(0,len(predict)):
    predict[i] = float(predict[i])
plt.plot(time, predict)
plt.show()
