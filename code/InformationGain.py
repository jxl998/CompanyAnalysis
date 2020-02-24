import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
# import feature
def label_delete(data, le):
    drop_list = []
    for col in data.columns:
        if col == "entname":
            continue
        temp_col = data[col]
        # print(data[col].count())
        if data[col].count() < 300:
            drop_list.append(col)
            continue
    for temp in drop_list:
        data = data.drop(temp, axis=1)
    for col in data.columns:
        temp_col = data[col]
        data[col] = le.fit_transform(data[col].astype(str))
        data[col] = data[col].astype(float)
    return data

class IG():
    def __init__(self, X, y):

        X = np.array(X)
        n_feature = np.shape(X)[1]
        n_y = len(y)

        orig_H = 0
        for i in set(y):
            # print(i)
            orig_H += -(y.count(i) / n_y) * math.log(y.count(i) / n_y)

        condi_H_list = []
        for i in range(n_feature):
            # print(str(i) + "================" )
            feature = X[:, i]
            sourted_feature = sorted(feature)
            threshold = [(sourted_feature[inde - 1] + sourted_feature[inde]) / 2 for inde in range(len(feature)) if
                         inde != 0]

            thre_set = set(threshold)
            if float(max(feature)) in thre_set:
                thre_set.remove(float(max(feature)))
            if min(feature) in thre_set:
                thre_set.remove(min(feature))
            pre_H = 0
            for thre in thre_set:
                lower = [y[s] for s in range(len(feature)) if feature[s] < thre]
                highter = [y[s] for s in range(len(feature)) if feature[s] > thre]
                H_l = 0
                for l in set(lower):
                    H_l += -(lower.count(l) / len(lower)) * math.log(lower.count(l) / len(lower))
                H_h = 0
                for h in set(highter):
                    H_h += -(highter.count(h) / len(highter)) * math.log(highter.count(h) / len(highter))
                temp_condi_H = len(lower) / n_y * H_l + len(highter) / n_y * H_h
                condi_H = orig_H - temp_condi_H
                pre_H = max(pre_H, condi_H)
            condi_H_list.append(pre_H)

        self.IG = condi_H_list

    def getIG(self):
        return self.IG
    # 返回列表中数值最小的索引
    def find_min_index(self):
        minimal_number = sys.float_info.max
        index = 0
        for i in range(len(self.IG)):
            if self.IG[i] <= minimal_number:
                minimal_number = self.IG[i]
                index = i
        return index


if __name__ == "__main__":
    X = [[5, 0, 0, 1],
         [6, 10, 1, 1],
         [7, 0, 1, 0]]
    Y = [0, 0, 1]
    # data = pd.read_csv('./data/company_data.csv', low_memory=False)
    #
    # le = LabelEncoder()
    # label_data = label_delete(data, le)
    # label_data = label_data.iloc[:, 1:]
    # X = label_data.values
    # B = X.tolist()
    # Y = []
    # for i in range(X.shape[0]):
    #     Y.append(i)
    # print(IG(X, Y).getIG())
    # print(IG(X, Y).find_min_index())