from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import utils

x, y = utils.load_data()
for k in range(3):
    x, y = shuffle(x, y)
    x_train, x_eva, y_train, y_eva = train_test_split(x, y, test_size=0.1)
    x_train, y_train = utils.balance_data(x_train, y_train, ratio=(0.5, 0.5))
    clf = RandomForestClassifier(random_state=0)
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
# Correct predictions (10% evaluation data): 0.9796708615682478, 0.46
# Correct predictions (10% evaluation data): 0.9749276759884281, 0.5204081632653061
# Correct predictions (10% evaluation data): 0.9798850574712644, 0.5185185185185185

# Ratio 6:4
# Correct predictions (10% evaluation data): 0.9003868471953579, 0.8442211055276382
# Correct predictions (10% evaluation data): 0.8970873786407767, 0.8029556650246306
# Correct predictions (10% evaluation data): 0.8819047619047619, 0.7868852459016393

# Ratio 1:1
# Correct predictions (10% evaluation data): 0.8720152817574021, 0.8064516129032258
# Correct predictions (10% evaluation data): 0.8440545808966862, 0.8599033816425121
# Correct predictions (10% evaluation data): 0.8576998050682261, 0.8599033816425121

# Ratio 4:6
# Correct predictions (10% evaluation data): 0.791907514450867, 0.9076923076923077
# Correct predictions (10% evaluation data): 0.802747791952895, 0.9112149532710281
# Correct predictions (10% evaluation data): 0.7955854126679462, 0.9319371727748691
