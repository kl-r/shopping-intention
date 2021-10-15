from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import utils

x, y = utils.load_data()
for k in range(3):
    x, y = shuffle(x, y)
    x_train, x_eva, y_train, y_eva = train_test_split(x, y, test_size=0.1)
    x_train, y_train = utils.balance_data(x_train, y_train, ratio=(0.5, 0.5))
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_eva)
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
# Correct predictions (10% evaluation data): 0.7884615384615384, 0.8238341968911918
# Correct predictions (10% evaluation data): 0.8091603053435115, 0.772972972972973
# Correct predictions (10% evaluation data): 0.7930056710775047, 0.76
