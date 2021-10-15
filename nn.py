import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import utils

x, y = utils.load_data()
x, y = utils.balance_data(x, y, ratio=(0.5, 0.5))
x, y = shuffle(x, y)
x_train, x_eva, y_train, y_eva = train_test_split(x, y, test_size=0.1)

# defining model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=x[0].shape),
    tf.keras.layers.Dense(units=32, activation="relu"),  # 3. feature extraction + 4. feature mapping
    tf.keras.layers.Dense(units=16, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=8, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=2, activation="sigmoid")  # 5. classification
])
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])


def lr_rate_schedule(epochs, lr):
    if epochs < 1000:
        return lr
    else:
        return lr*0.1
    return lr


# model training
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_rate_schedule)  # variable lr
log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # training graphs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
print(len(x_train), len(x_train[0]), len(y_train), len(y_train[0]))

# requires np.asarray or throws "Data cardinality is ambiguous"
model.fit(np.asarray(x_train), np.asarray(y_train), epochs=1000, batch_size=16, verbose=1,
          callbacks=[lr_callback, tensorboard_callback])

# model evaluation
predictions = model.predict(np.asarray(x_eva))
eva_score = [0, 0]
eva_len = [0, 0]
for i in range(len(predictions)):
    if np.argmax(predictions[i]) == np.argmax(y_eva[i]):
        eva_score[np.argmax(predictions[i])] += 1
    if np.argmax(y_eva[i]) == 0:
        eva_len[0] += 1
    else:
        eva_len[1] += 1
print("Correct predictions (10% evaluation data): "+str(eva_score[0]/eva_len[0])+", "+str(eva_score[1]/eva_len[1]))
# Correct predictions (10% evaluation data): 0.9514747859181731, 0.554945054945055
#
#

# Ratio 6:4
# Correct predictions (10% evaluation data): 0.8298611111111112, 0.8465608465608465
# Correct predictions (10% evaluation data): 0.8214285714285714, 0.8578680203045685
# Correct predictions (10% evaluation data): 0.8125, 0.8518518518518519

# Ratio 1:1
# Correct predictions (10% evaluation data): 0.7978723404255319, 0.8711340206185567
# Correct predictions (10% evaluation data): 0.7315789473684211, 0.9427083333333334
# Correct predictions (10% evaluation data): 0.8177083333333334, 0.8473684210526315

# Ratio 4:6
# Correct predictions (10% evaluation data): 0.6774193548387096, 0.9329896907216495
#
#
