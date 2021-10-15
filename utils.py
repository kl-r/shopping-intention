import csv
import numpy as np
from sklearn.utils import shuffle


def load_data():  # one-hot
    month = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
             'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    visitor = {'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2}
    weekend = {'FALSE': 0, 'TRUE': 1}
    revenue = {'FALSE': [1, 0], 'TRUE': [0, 1]}

    with open("online_shoppers_intention.csv", newline='') as read:
        reader = csv.reader(read)
        data = list(reader)

    names = data.pop(0)
    x = np.zeros((len(data), 75))
    y = np.zeros((len(data), 2))

    for i in range(len(data)):
        d = data[i]
        v = []
    # Administrative,Administrative_Duration, Informational,Informational_Duration,
    # ProductRelated,ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay
        for k in range(9):
            v.append(float(d[k]))
    # Month
        m = np.zeros(12)
        m[month[d[10]]] = 1
        v.extend(m)
    # OperatingSystem
        o = np.zeros(8)
        o[int(d[11])-1] = 1
        v.extend(o)
    # Browser
        b = np.zeros(13)
        b[int(d[12])-1] = 1
        v.extend(b)
    # Region
        r = np.zeros(9)
        r[int(d[13])-1] = 1
        v.extend(r)
    # TrafficType
        t = np.zeros(20)
        t[int(d[14])-1] = 1
        v.extend(t)
    # VisitorType
        vt = np.zeros(3)
        vt[visitor[d[15]]] = 1
        v.extend(vt)
    # Weekend
        v.append(weekend[d[16]])
        x[i] = v.copy()
    # Revenue
        y[i] = revenue[d[17]]
    return x, y


def separate_data(x, y):
    x_1, x_2 = [], []
    y_1, y_2 = [], []
    for i in range(len(x)):
        if y[i][0] == 1:
            x_1.append(x[i])
            y_1.append(y[i])
        else:
            x_2.append(x[i])
            y_2.append(y[i])
    return x_1, x_2, y_1, y_2


def balance_data(x, y, ratio=(0.6, 0.4), randomize=True):
    x_1, x_2, y_1, y_2 = separate_data(x, y)
    x_num = int(len(x_2)/ratio[1]*ratio[0])
    if randomize:
        x_1, y_1 = shuffle(x_1, y_1)
    x_new = x_1[0:x_num]
    x_new.extend(x_2)
    y_new = y_1[0:x_num]
    y_new.extend(y_2)
    return x_new, y_new
