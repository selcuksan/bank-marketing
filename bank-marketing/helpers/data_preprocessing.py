import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor


##################################
# 2. Data Preprocessing
##################################


def replace_with_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def bool_to_int(dataframe):
    bool_cols = [
        col for col in dataframe.columns if dataframe[col].dtypes == "bool"]
    for col in bool_cols:
        dataframe[col] = dataframe[col].astype(int)


def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe


def binary_columns(dataframe):
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in ["int", "float"]
                   and dataframe[col].nunique() == 2]
    return binary_cols


def rare_encoder(dataframe, rare_perc=0.01):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(
            rare_labels), 'Rare', temp_df[var])

    return temp_df


def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def local_outlier_factor(dataframe, num_cols, n_neighbors=20, plot=True):
    """
    Parameters
    ----------
    dataframe
    num_cols
    n_neighbors
    plot

    Returns
        Çok değişkenli aykırı değer analizinde aykırı olan
        gözlemlerin index değerlerini döndürür.
    -------

    """
    clf = LocalOutlierFactor(n_neighbors)
    clf.fit_predict(dataframe[num_cols])
    df_scores = clf.negative_outlier_factor_
    if plot:
        scores = pd.DataFrame(np.sort(df_scores))
        scores.plot(stacked=True, xlim=[0, 20], style=".-")
        plt.show()
    th = np.sort(df_scores)[3]
    return dataframe[df_scores < th].index


def remove_outlier(dataframe, col_name, q1=0.01, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    df_without_outliers = dataframe[~(
            (dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers
