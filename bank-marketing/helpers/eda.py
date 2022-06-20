import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from helpers.data_preprocessing import outlier_thresholds


################################################
# 1. Exploratory Data Analysis
################################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False, bins=20):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40,
                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=bins)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_num(dataframe, target, numerical_col, target_type="num"):
    if target_type == "num":
        try:
            print(dataframe.groupby(pd.qcut(x=dataframe[target], q=4)).agg({numerical_col: "mean"}), end="\n\n\n")
        except ValueError:
            print(
                dataframe.groupby(pd.qcut(x=dataframe[target].rank(method="first"), q=4)).agg({numerical_col: "mean"}),
                end="\n\n\n")
    else:
        print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col, target_type="num"):
    if target_type == "num":
        print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(
            categorical_col)[target].mean()}), end="\n\n\n")
    # else:
    #     print(pd.DataFrame({categorical_col + "_COUNT": dataframe.groupby(categorical_col)[target].count()}),
    #           end="\n\n\n")


def correlation_matrix(dataframe, cols):
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show(block=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns
                 if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr_matrix, cmap="RdBu")
        plt.show()
    return drop_list


def drop_high_correlated_cols(dataframe, drop_cols, plot=False):
    dataframe = dataframe.drop(drop_cols, axis=1)
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr_matrix, cmap="RdBu")
        plt.show()
    return dataframe


def missing_values_table(dataframe, na_name=True):
    na_columns = [
        col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum()
    ratio = (dataframe[na_columns].isnull().sum().sort_values(ascending=False) / dataframe.shape[0]).sort_values(
        ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)
    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
        na_flag_df = temp_df.loc[:, temp_df.columns.str.contains("_NA_")]
    for col in na_flag_df:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby([col])[target].mean(),
                            "COUNT": temp_df.groupby([col])[target].count()}), end="\n\n\n")


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({
            "COUNT": dataframe[col].value_counts(),
            "RATIO": dataframe[col].value_counts() / dataframe.shape[0],
            "TARGET_MEAN": dataframe.groupby([col])[target].mean()
        }), end="\n\n\n")


def grab_outliers(dataframe, col_name, index=False, plot=False, q1=0.01, q3=0.99):
    if plot:
        sns.boxplot(x=dataframe[col_name])
        plt.show()
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low)
                         | (dataframe[col_name] > up))].head(5))
    else:
        print(dataframe[((dataframe[col_name] < low)
                         | (dataframe[col_name] > up))])
    if index:
        outlier_index = dataframe[(
                (dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index
