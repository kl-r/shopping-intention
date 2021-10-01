import csv
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

month = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
         'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
visitor = {'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2}
weekend = {'FALSE': 0, 'TRUE': 1}
revenue = {'FALSE': [1, 0], 'TRUE': [0, 1]}


def lr_rate_schedule(epochs, lr):
    if epochs < 1000:
        return lr
    else:
        return lr*0.1
    return lr


# 1. raw data
with open("online_shoppers_intention.csv", newline='') as read:
    reader = csv.reader(read)
    data = list(reader)

names = data.pop(0)
x = np.zeros((len(data), len(data[0])-1+11+2))
y = np.zeros((len(data), 2))

# 2. preprocessing
for i in range(len(data)):
    d = data[i]
    for k in range(9):
        x[i][k] = float(d[k])
    for k in range(10, 14):
        x[i][k] = float(d[k+1])
    x[i][15] = weekend[d[16]]
    x[i][month[d[10]]+14] = 1
    x[i][visitor[d[15]]+14+12] = 1
    y[i] = revenue[d[17]]

x, y = shuffle(x, y)
x_train, x_eva, y_train, y_eva = train_test_split(x, y, test_size=0.1)

# defining model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=x[0].shape),
    tf.keras.layers.Dense(units=16, activation="relu"),  # 3. feature extraction + 4. feature mapping
    tf.keras.layers.Dense(units=16, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=2, activation="softmax")  # 5. classification
])
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

# model training
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_rate_schedule)  # variable lr
log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # training graphs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(x_train, y_train, epochs=2000, batch_size=64, verbose=1, validation_split=0.05,
          callbacks=[lr_callback, tensorboard_callback])

# model evaluation
predictions = model.predict(x_eva)
eva_score = [0, 0]
eva_len = [0, 0]
for i in range(len(predictions)):
    #print(predictions[i], y_eva[i])
    if np.argmax(predictions[i]) == np.argmax(y_eva[i]):
        eva_score[np.argmax(predictions[i])] += 1
    if np.argmax(y_eva[i]) == 0:
        eva_len[0] += 1
    else:
        eva_len[1] += 1
print("Correct predictions (10% evaluation data): "+str(eva_score[0]/eva_len[0])+", "+str(eva_score[1]/eva_len[1]))