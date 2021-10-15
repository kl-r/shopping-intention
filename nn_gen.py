import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import utils

batch_size = 8
ratio = 0.5
x, y = utils.load_data()
x, y = shuffle(x, y)
x_train, x_eva, y_train, y_eva = train_test_split(x, y, test_size=0.1)
x1, x2, y1, y2 = utils.separate_data(x_train, y_train)

# defining model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=x[0].shape),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=2, activation="sigmoid")
])
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])


def lr_rate_schedule(epochs, lr):
    if epochs < 1000:
        return lr
    else:
        return lr*0.1
    return lr


class BalancedGenerator(tf.keras.utils.Sequence):
    def __init__(self, x1, x2, y1, y2, batch_size, ratio):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.batch_size = batch_size
        self.ratio = ratio

    def __len__(self):
        return (np.ceil((len(self.x1) + len(self.x2)) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.x1[idx * int(self.batch_size * self.ratio): (idx + 1) * int(self.batch_size * self.ratio)]
        batch_x.extend(self.x2[idx * int(self.batch_size - self.batch_size * self.ratio): (idx + 1) * int(self.batch_size - self.batch_size * self.ratio)])
        batch_y = self.y1[idx * int(self.batch_size * self.ratio): (idx + 1) * int(self.batch_size * self.ratio)]
        batch_y.extend(self.y2[idx * int(self.batch_size - self.batch_size * self.ratio): (idx + 1) * int(self.batch_size - self.batch_size * self.ratio)])
        return np.array(batch_x), np.array(batch_y)


# model training
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_rate_schedule)  # variable lr
log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # training graphs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
print(len(x_train), len(x_train[0]), len(y_train), len(y_train[0]))

training_batch_generator = BalancedGenerator(x1, x2, y1, y2, batch_size, ratio)
# requires np.asarray or throws "Data cardinality is ambiguous"
model.fit(training_batch_generator, epochs=1000, verbose=1, callbacks=[lr_callback, tensorboard_callback])

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

# Ratio 1:1
# Correct predictions (10% evaluation data): 0.9191246431969553, 0.7142857142857143
# Correct predictions (10% evaluation data): 0.9164319248826291, 0.6726190476190477
# Larger arch: Correct predictions (10% evaluation data): 0.9580078125, 0.5167464114832536

# Ratio 4:6
# Correct predictions (10% evaluation data): 0.9689922480620154, 0.4079601990049751

# Ratio 4:6
# Correct predictions (10% evaluation data): 0.9586538461538462, 0.5699481865284974


# Correct predictions (10% evaluation data): 0.9496124031007752, 0.572139303482587 no tanh