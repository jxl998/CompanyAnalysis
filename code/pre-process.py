#-*- coding : utf-8-*-
# coding:unicode_escape

import glob
import pandas as pd
import numpy as np
import os
from sklearn.datasets.base import Bunch

fl1 = ["ent_onlineshop.csv", "ent_branch.csv", "ent_contribution_year.csv", "ent_contribution.csv"]
fl2 = ["ent_investment.csv", "recruit_qcwy.csv", "recruit_zhyc.csv", "recruit_zlzp.csv", "intangible_brand.csv",
       "intangible_copyright.csv", "intangible_patent.csv", "jn_tech_center.csv"]
fl3 = ["ent_bid.csv"]
fl4_1 = ["ent_guarantee.csv", "justice_declare.csv", "justice_enforced.csv", "justice_judge_new.csv"]
fl4_2 = ["business_risk_abnormal.csv", "business_risk_all_punish.csv", "business_risk_rightpledge.csv",
         "business_risk_taxunpaid.csv", "administrative_punishment.csv"]
fl4_3 = ["ent_guarantee.csv", "exception_list.csv", "enterprise_keep_contract.csv", "jn_credit_info.csv", "justice_credit_aic.csv"]
fl5 = ["trademark_infoa.csv", "trademark_infob.csv"]


def select_file(file_list):
    df_list = []
    for c in file_list:
        df = pd.read_csv(str(c), encoding='gbk')
        df['entname'] = df.entname.astype(np.str)
        df = df.set_index('entname')
        df = df.loc[df.index.drop_duplicates(keep=False), :]  # 去掉df1中重复索引
        df_list.append(df)
    result = pd.concat(df_list, axis=1, join='outer')
    return result






def pre():
    csv_list = glob.glob('Data_FCDS_hashed/*.csv')
    csv_list.remove("Data_FCDS_hashed/company_baseinfo.csv")
    print(csv_list)
    df_list = []
    for c in csv_list:
        df = pd.read_csv(str(c), encoding='gbk')
        df['entname'] = df.entname.astype(np.str)
        df = df.set_index('entname')
        df = df.loc[df.index.drop_duplicates(keep=False), :]  # 去掉df1中重复索引
        df_list.append(df)
    result = pd.concat(df_list, axis=1, join='outer')
    result.to_csv("company_data.csv", index=True, encoding='utf_8_sig')