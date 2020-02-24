import sys
# sys.path.append("E:\\python文件\\CompanyAnalysis\\code\\preprocess.py")
# import preprocess as pre
import InformationGain
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import itertools


# d.create_meta()
# is_number判断传输的字符是浮点数还是其他字符，暂时没用
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

# 将非数值数值处理为数值信息，并对每一列数值数量少于300的特征删除
# 接收参数 data为dataframe类型，le为LabelEncoder()
def prepare_data(data, le):
    drop_list = []
    # for col in data.columns:

        # if col == "entname":
        #     continue
        # temp_col = data[col]
        # print(data[col].count())
        # if data[col].count() < 300:
        #     drop_list.append(col)
        #     continue
    # for temp in drop_list:
    #     data = data.drop(temp, axis=1)
    for col in data.columns:
        temp_col = data[col]
        data[col] = le.fit_transform(data[col].astype(str))
        data[col] = data[col].astype(float)
    data = data.drop("entname", axis=1)
    return data
# GMM混合高斯模型参数选择，matplotlib绘制图像
# 对GMM模型用几个高斯分布来拟合分布，详情见https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
# 接受data为dataframe类型
def GMM_parameter_selection(data):
    data = data.iloc[:, 1:]
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'full']
    best_gmm = None
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            # print(n_components)
            gmm = GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type, random_state=42)
            gmm.fit(data)
            bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    # spl.title('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    # splot = plt.subplot(2, 1, 2)
    # Y_ = clf.predict(data)
    # for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
    #                                            color_iter)):
    #     v, w = linalg.eigh(cov)
    #     if not np.any(Y_ == i):
    #         continue
    #     plt.scatter(data[Y_ == i, 0], data[Y_ == i, 1], .8, color=color)
    #
    #     Plot an ellipse to show the Gaussian component
        # angle = np.arctan2(w[0][1], w[0][0])
        # angle = 180. * angle / np.pi  # convert to degrees
        # v = 2. * np.sqrt(2.) * np.sqrt(v)
        # ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        # ell.set_clip_box(splot.bbox)
        # ell.set_alpha(.5)
        # splot.add_artist(ell)

    # plt.xticks(())
    # plt.yticks(())
    plt.title('Number of components')
    # plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()




    # gmm = GaussianMixture(n_components=10)
    # plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
    # plt.plot(n_components, [m.aic(data) for m in models], label='AIC')
    # plt.legend(loc='best')

    # gmm.fit(data)
    # labels = gmm.predict(data)
    # print(labels)
    # data = data.iloc[:, ::-1]
    # plt.scatter(data[:, 0], data[:, 1], c=labels, s=40, cmap='viridis')
    #
    # metrics.calinski_harabaz_score(X, labels)
# GMM混合高斯模型，进行训练并输出标签
# 标签为1,2,3,4,5的数字，接收参数data为dataframe类型
def GMM_model(data):
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
    gmm.fit(data)
    labels = gmm.predict(data)
    score = metrics.silhouette_score(data, labels)
    # print(metrics.silhouette_score(data, labels, sample_size=len(data), metric='euclidean'))
    data['result'] = labels
    return [data, labels, score]

#特征提取，根据轮廓系数进行选择
# 轮廓系数越接近1越好
# InformationGain信息增益输出为每一列对应的权重，数值越小对于最终的聚类帮助越小
# data为[data, labels, score]类型，data[0]为训练数据，data[1]为标签，data[2]为轮廓系数
def extract(data):

    X = data[0].values
    Y = data[1]
    IG = InformationGain.IG(X, Y)
    # result = IG.getIG()
    column_name = list(data.columns)
    # min_number = sys.float_info.max
    # index = 0
    # for i in range(len(result)):
    #     if result[i] < min_number:
    #         min_number = result[i]
    #         index = i
    data.drop(column_name[IG.find_min_index()])
    return data


if __name__ == "__main__":
    # d = pre.DataProcess()
    # 加载拼接在一起的数据
    data = pd.read_csv('../data/company_data.csv', low_memory=False)
    le = LabelEncoder()
    label_data = prepare_data(data, le)
    # 画图像判断n_components
    GMM_parameter_selection(label_data)
    # 这里需要将score和删除的特征打印出来，逐步尝试
    data_labels_score = GMM_model(label_data)
    # while score < data_labels_score[2]:
    #     score = data_labels_score[2]
    #     print(score)
    #     extract_result = extract(data_labels_score)
    #     data_labels_score = GMM_model(extract_result)
    print("extract feature finish")
    # data_result = GMM_model(label_data)
    # discriminated_data = GMM_model(data_result)


