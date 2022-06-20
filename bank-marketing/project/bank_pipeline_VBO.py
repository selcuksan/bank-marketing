################################################
# End-to-End Bank Dataset Machine Learning Project Pipeline
################################################
import pandas as pd

from helpers.data_preprocessing import *
from helpers.eda import *
from helpers.model_validation import *
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


def create_date_features(dataframe):
    dataframe['month'] = dataframe.DATE.dt.month
    dataframe['day_of_month'] = dataframe.DATE.dt.day
    dataframe['day_of_year'] = dataframe.DATE.dt.dayofyear
    dataframe['week_of_year'] = dataframe.DATE.dt.weekofyear
    dataframe['day_of_week'] = dataframe.DATE.dt.dayofweek
    dataframe['year'] = dataframe.DATE.dt.year
    dataframe["is_wknd"] = dataframe.DATE.dt.weekday // 5
    dataframe['is_month_start'] = dataframe.DATE.dt.is_month_start.astype(int)
    dataframe['is_month_end'] = dataframe.DATE.dt.is_month_end.astype(int)
    print(dataframe["DATE"])
    dataframe = dataframe.drop("DATE", axis=1)
    return dataframe


def bank_data_prep(dataframe, random_user=None):
    if random_user is not None:
        dataframe = pd.concat([dataframe, random_user], ignore_index=False, axis=0)
        random_user_index = dataframe.tail(1).index[0]
        print(random_user_index)
    dataframe.columns = [col.upper() for col in dataframe.columns]
    dataframe["NEW_PDAYS"] = dataframe["PDAYS"].apply(lambda x: "No" if x == -1 else "Yes")
    dataframe["NEW_PREVIOUS"] = dataframe["PREVIOUS"].apply(lambda x: "No" if x == 0 else "Yes")
    dataframe["MONTH_NUM"] = dataframe["MONTH"].map({
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11,
        "dec": 12
    })
    dataframe["DATE"] = "2011-" + dataframe["MONTH_NUM"].astype(str) + "-" + dataframe["DAY"].astype(int).astype(str)
    dataframe["DATE"] = pd.to_datetime(dataframe["DATE"])
    dataframe = create_date_features(dataframe)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=10, car_th=20)

    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # local_outliers = local_outlier_factor(dataframe, num_cols, plot=False)
    # dataframe = dataframe.loc[~dataframe.index.isin(local_outliers)]
    random_user = dataframe[dataframe.index == random_user_index]
    dataframe = dataframe.drop(random_user_index)
    yes_df = dataframe.loc[dataframe["Y"] == "yes"].sample(5280)
    no_df = dataframe.loc[dataframe["Y"] == "no"].sample(5280)
    df = pd.concat([no_df, yes_df], axis=0, ignore_index=False)
    df = pd.concat([df, random_user], ignore_index=False, axis=0)
    Y = df["Y"]
    X = df.drop("Y", axis=1)

    if random_user is not None:
        cat_cols, num_cols, cat_but_car = grab_col_names(X, cat_th=10, car_th=20)
        X_encoded = one_hot_encoder(X, cat_cols, drop_first=True)
        # return X_encoded[X_encoded.index == random_user_index]
        return X_encoded.tail(1)

    cat_cols, num_cols, cat_but_car = grab_col_names(X, cat_th=10, car_th=20)
    X_encoded = one_hot_encoder(X, cat_cols, drop_first=True)
    Y_encoded = pd.Series(LabelEncoder().fit_transform(Y))

    return X_encoded, Y_encoded


def feature_extraction(dataframe, random_user=None):
    dataframe = pd.concat([dataframe, random_user], ignore_index=False, axis=0)
    dataframe.columns = [col.upper() for col in dataframe.columns]
    dataframe["NEW_PDAYS"] = dataframe["PDAYS"].apply(lambda x: "No" if x == -1 else "Yes")
    dataframe["NEW_PREVIOUS"] = dataframe["PREVIOUS"].apply(lambda x: "No" if x == 0 else "Yes")
    dataframe["MONTH_NUM"] = dataframe["MONTH"].map({
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11,
        "dec": 12
    })
    dataframe["DATE"] = "2011-" + dataframe["MONTH_NUM"].astype(str) + "-" + dataframe["DAY"].astype(int).astype(str)
    dataframe["DATE"] = pd.to_datetime(dataframe["DATE"])
    dataframe = create_date_features(dataframe)
    dataframe = dataframe.drop("DATE", axis=1)


knn_params = {"n_neighbors": range(2, 45)}

cart_params = {'max_depth': range(1, 21),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300, 750, 1000]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [500, 1000],
                   "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False,
                                         eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


def main(classifiers):
    bank_full = pd.read_csv("Veri_madenciligi_proje/bank-full.csv", delimiter=";")
    df = bank_full.copy()
    X, y = bank_data_prep(df)
    base_models_clf(X, y, scoring=["f1"])
    best_models = hyperparameter_optimization_clf(X, y, classifiers=classifiers, scoring=["f1"])
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf


if __name__ == "__main__":
    print("İşlem başladı")
    main(classifiers)
