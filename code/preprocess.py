import pandas as pd
import os
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder

import json

# *************** Preparing Data ***************
# 预处理数据： 多种数据类型：？？   categorical features：one hot 或 label encoder
# data = pd.read_csv('data/company_data.csv', dtype={'taxunpaidnum':float})
data = pd.read_csv('../data/company_data.csv', low_memory=False)
data.head()
remove_list = ["taxunpaidnum", "iftopub", "xzbz", "xzbzmc", "cbztmc", ]


# *************** Data quality report ***************
def quality_report():
    # print(data.info())
    print('\n*************** data quality report ***************')
    print('continuous features:')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data.describe())
    print('\ncategorical features:')
    for col in data.columns:
        if data[col].dtype == "object":
            print("For column {} count is {} cardinality is: {}".format(col, data[col].count(), data[col].nunique()))


# *************** Create meta ***************
def create_meta():
    meta = {
        'feature_names': list(data.columns),
        'categorical_features': {
            column: list(data[column].unique())
            for column in data.columns
            if data[column].dtype == 'object'
        },
    }

    # 对数据特征的删除处理在以下完成
    # 删除irregular cardinality特征
    meta['feature_names'].remove('entname')
    meta['categorical_features'].pop('entname')

    # 删除count<200的特征
    for col in data.columns:
        if data[col].count() < 200:
            meta['feature_names'].remove(col)
            print("Delete feature:{}".format(col))
            if data[col].dtype == 'object':
                meta['categorical_features'].pop(col)
    with open('../data/company_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


#对categorical_features进行LabelEncoder处理
def sovle_cf():
    with open(os.path.join('../data/company_meta.json'), 'r') as f:
        meta = json.load(f)
    cf = meta['categorical_features']
    le = LabelEncoder()
    for i in cf:
        data[i] = le.fit_transform(data[i].astype(str))
    return data


# *************** Load data ***************
def load_data():
    with open(os.path.join('../data/company_meta.json'), 'r') as f:
        meta = json.load(f)
    names = meta['feature_names']
    #这行代码报错
    # train = pd.read_csv('../data/company_data.csv', usecols=names, engine='python')
    return Bunch(
        # data=train[names],
        feature_names=meta['feature_names'],
        categorical_features=meta['categorical_features'],
    )


if __name__ == "__main__":
    quality_report()
    if not os.path.exists('../data/company_meta.json'):
        print('\n*************** preparing meta ***************')
        create_meta()

    print('\n*************** load data ***************')
    dataset = load_data()


    # 接下来都用dataset 比如 dataset.data dataset.feature_names dataset.categorical_features
