from sklearn.model_selection import cross_validate, validation_curve, GridSearchCV
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier, RandomForestRegressor, VotingClassifier, AdaBoostClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


def cross_validate_classification(model, X, y, cv_num=5):
    cv_results = cross_validate(model, X, y, cv=cv_num, scoring=[
        "accuracy", "f1", "roc_auc"])
    print("test_accuracy: ", cv_results["test_accuracy"].mean())
    print("test_f1: ", cv_results["test_f1"].mean())
    print("test_roc_auc: ", cv_results["test_roc_auc"].mean())


def cross_validate_regression(model, X, y, cv_num=5):
    cv_results = cross_validate(model, X, y, cv=cv_num, scoring=[
        "neg_root_mean_squared_error"])
    print("rmse: ", -cv_results["test_neg_root_mean_squared_error"].mean())


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number Of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame(
        {'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(
        by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('Importances.png')


def base_models_reg(X, y, scoring="neg_root_mean_squared_error"):
    print("Base Models....")
    regressors = [('LR', LinearRegression()),
                  ('KNN', KNeighborsRegressor()),
                  ("SVR", SVR()),
                  ("CART", DecisionTreeRegressor()),
                  ("RF", RandomForestRegressor()),
                  ('Adaboost', AdaBoostRegressor()),
                  ('GBM', GradientBoostingRegressor()),
                  ('XGBoost', XGBRegressor(
                      use_label_encoder=False)),
                  ('LightGBM', LGBMRegressor())
                  ]

    for name, regressor in regressors:
        cv_results = cross_validate(regressor, X, y, cv=3, scoring=scoring)
        print("***************HATA VERECEK************")
        print(
            f"{scoring}: {round(-cv_results['test_neg_root_mean_squared_error'].mean(), 4)} ({name}) ")


def base_models_clf(X, y, scoring):
    print("Base Models...")
    classifiers = [
        ('LR', LogisticRegression(max_iter=10000)),
        ('KNN', KNeighborsClassifier()),
        ("SVC", SVC()),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier()),
        ('Adaboost', AdaBoostClassifier()),
        ('GBM', GradientBoostingClassifier()),
        ('XGBoost', XGBClassifier(
            use_label_encoder=False, eval_metric='logloss')),
        ('LightGBM', LGBMClassifier()),
        # ('CatBoost', CatBoostClassifier(verbose=False))
    ]
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        for score in scoring:
            print(f"{score}: {round(cv_results['test_' + score].mean(), 4)} ({name}) ")


def hyperparameter_optimization_clf(X, y, classifiers, scoring, cv=3):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        for score in scoring:
            print(f"{score}: {round(cv_results['test_' + score].mean(), 4)} ({name}) ")
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        for score in scoring:
            print(f"{score}: {round(cv_results['test_' + score].mean(), 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


def hyperparameter_optimization_reg(X, y, regressors, cv=3, scoring="neg_root_mean_squared_error"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(cv_results)
        print("***************HATA VERECEK************")
        print(
            f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv,
                               n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(
            f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=[
        "accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


def voting_regressor(best_models, X, y):
    print("Voting Classifier...")
    voting_reg = VotingRegressor(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                             ('LightGBM', best_models["LightGBM"])],
                                 voting='soft').fit(X, y)
    cv_results = cross_validate(voting_reg, X, y, cv=3, scoring=[
        "neg_root_mean_squared_error"])
    print(cv_results)
    print("***************HATA VERECEK************")
    print(f"RMSE: {-cv_results['test_neg_root_mean_squared_error'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    return voting_reg


"""knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300, 750]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500, 100],
                   "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False,
                eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]
"""
