import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import utils

batch_size = 16
ratio = 0.5
x, y = utils.load_data()
x, y = shuffle(x, y)
x_train, x_eva, y_train, y_eva = train_test_split(x, y, test_size=0.1)
x1, x2, y1, y2 = utils.separate_data(x_train, y_train)

# defining model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=x[0].shape),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=16, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=8, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=2, activation="softmax")
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
        batch_x.extend(self.x2[idx * int(self.batch_size * self.ratio): (idx + 1) * int(self.batch_size * self.ratio)])
        batch_y = self.y1[idx * int(self.batch_size * self.ratio): (idx + 1) * int(self.batch_size * self.ratio)]
        batch_y.extend(self.y2[idx * int(self.batch_size * self.ratio): (idx + 1) * int(self.batch_size * self.ratio)])
        return np.array(batch_x), np.array(batch_y)


# model training
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_rate_schedule)  # variable lr
log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # training graphs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

training_batch_generator = BalancedGenerator(x1, x2, y1, y2, batch_size, ratio)

for k in range(5):  # control overfit
    # requires np.asarray or throws "Data cardinality is ambiguous"
    model.fit(training_batch_generator, epochs=50, verbose=1, callbacks=[lr_callback, tensorboard_callback])

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

# Ratio 4:6
# Correct predictions (10% evaluation data): 0.9689922480620154, 0.4079601990049751

# Ratio 4:6
# Correct predictions (10% evaluation data): 0.9586538461538462, 0.5699481865284974


# Larger arch: Correct predictions (10% evaluation data): 0.9580078125, 0.5167464114832536
# no tanh: Correct predictions (10% evaluation data): 0.9496124031007752, 0.572139303482587
# 32 batch: Correct predictions (10% evaluation data): 0.9411187438665358, 0.5186915887850467
# GELU: Correct predictions (10% evaluation data): 0.9922254616132167, 0.28921568627450983
# SELU: Correct predictions (10% evaluation data): 0.9754716981132076, 0.45664739884393063
# SELU #2: Correct predictions (10% evaluation data): 0.9771863117870723, 0.35359116022099446

# RELU 32-16-8, batch_size = 16, bin crossentropy: 0.9770334928229665, 0.39893617021276595
# cat crossentropy: Correct predictions (10% evaluation data): 0.9924599434495759, 0.12790697674418605
# mse: Correct predictions (10% evaluation data): 0.9526570048309179, 0.5505050505050505
# softmax: Correct predictions (10% evaluation data): 0.9444985394352483, 0.6553398058252428
# evaluation every 100 epochs:
# Correct predictions (10% evaluation data): 0.9622823984526112, 0.48743718592964824
# Correct predictions (10% evaluation data): 0.9584139264990329, 0.5326633165829145
# Correct predictions (10% evaluation data): 0.9177949709864603, 0.6633165829145728
# Correct predictions (10% evaluation data): 0.9671179883945842, 0.47738693467336685
# Correct predictions (10% evaluation data): 0.9661508704061895, 0.4623115577889447
# Correct predictions (10% evaluation data): 0.9632495164410058, 0.4472361809045226
# Correct predictions (10% evaluation data): 0.971953578336557, 0.45226130653266333
# Correct predictions (10% evaluation data): 0.9690522243713733, 0.41708542713567837
# 100 epochs, dropouts 0.1: Correct predictions (10% evaluation data): 0.9415708812260536, 0.6825396825396826
