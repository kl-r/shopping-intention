from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import utils

x, y = utils.load_data()
for k in range(3):
    x, y = shuffle(x, y)
    x_train, x_eva, y_train, y_eva = train_test_split(x, y, test_size=0.1)
    x_train, y_train = utils.balance_data(x_train, y_train, ratio=(0.4, 0.6))

    knn = KNeighborsClassifier(n_neighbors=2, weights='distance')
    knn.fit(x_train, y_train)

    predictions = knn.predict(x_eva)
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
# Correct predictions (10% evaluation data): 0.9019980970504282, 0.3791208791208791
# Correct predictions (10% evaluation data): 0.9133782483156881, 0.4587628865979381
# Correct predictions (10% evaluation data): 0.9168260038240917, 0.40641711229946526

# For ratio 6:4
# Correct predictions (10% evaluation data): 0.7096466093600764, 0.6881720430107527
# Correct predictions (10% evaluation data): 0.7256038647342995, 0.6414141414141414
# Correct predictions (10% evaluation data): 0.6986817325800376, 0.6842105263157895

# For ratio 1:1
# Correct predictions (10% evaluation data): 0.712707182320442, 0.6666666666666666
# Correct predictions (10% evaluation data): 0.65, 0.6758241758241759
# Correct predictions (10% evaluation data): 0.6386138613861386, 0.6444444444444445

# For ratio 6:4
# Correct predictions (10% evaluation data): 0.6501429933269781, 0.7445652173913043
# Correct predictions (10% evaluation data): 0.6264150943396226, 0.7456647398843931
# Correct predictions (10% evaluation data): 0.6337209302325582, 0.7611940298507462
