################################################
# End-to-End Bank Dataset Machine Learning Project
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

import matplotlib.pyplot as plt
from helpers.data_preprocessing import *
from helpers.eda import *
from helpers.model_validation import *
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, mannwhitneyu

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

##################################
# Getting and Merging Data
##################################

bank_full = pd.read_csv("Veri_madenciligi_proje/bank-full.csv", delimiter=";")
df = bank_full.copy()

################################################
# 1. Exploratory Data Analysis
################################################
check_df(df)

# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

#############################################################################################
# Data analysis
# df.columns = ['age', 'job', 'marital', 'education', 'default',
# 'balance', 'housing', 'loan', 'contact', 'day',
# 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
plt.figure(figsize=(10, 10))

# AGE
sns.distplot(df["age"])
sns.boxplot(x="y", y="age", data=df)
# plt.show()
# JOB
sns.countplot(y="job", hue="y", data=df)
plt.pie(df["job"].value_counts(), labels=df["job"].value_counts().index, autopct='%1.1f%%')
# plt.show()
# MARITAL
sns.countplot(y="marital", hue="y", data=df)
explode = (0, 0.1, 0)
plt.pie(df["marital"].value_counts(), explode=explode,
        labels=df["marital"].value_counts().index, autopct='%1.1f%%')
# plt.show()
# EDUCATION
sns.countplot(y="education", hue="y", data=df)
sns.countplot(y="job", hue="education", data=df)
# plt.show()
# BALANCE
sns.histplot(x="education", y="balance", hue="y", data=df)
sns.scatterplot(x="age", y="balance", hue="y", data=df)
sns.histplot(x="balance", hue="y", data=df)
# plt.show()
# DENSITY
sns.histplot(x="duration", hue="y", data=df)
# plt.show()
################################################
# 1. EDA (AB Testing)
################################################
#######################
# Normallik Varsayımı
#######################
# H0: M1 = M2   -   Normal Dağılım varsayımı sağlanmaktadır.
# H1: M1 != M2  -   Sağlanmamaktadır.

test_stat, pvalue = shapiro(df.loc[df["y"] == "yes", "balance"])
print(f"test_stat: {test_stat}, pvalue: {pvalue}")
test_stat, pvalue = shapiro(df.loc[df["y"] == "no", "balance"].dropna())
print(f"test_stat: {test_stat}, pvalue: {pvalue}")
# Normallik varsayımı sağlanmamaktadır.

# non parametrik mann whitney u testi uygulanması
test_stat, pvalue = mannwhitneyu(df.loc[df["y"] == "yes", "balance"],
                                 df.loc[df["y"] == "no", "balance"])
print(f"test_stat: {test_stat}, pvalue: {pvalue}")
# p-value < 0.05
# H0 hipotezini reddedemeyiz.
# ist. ol. %95 güven düzeyinde iki grup ortalamaları arasında anlamlı fark yoktur.
#############################################################################################

# Kategorik değişken analizi
for col in cat_cols:
    cat_summary(df, col, False)

sns.countplot(x="month", data=df,
              order=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
# plt.show()

# Sayisal değişken analizi
for col in num_cols:
    num_summary(df, col, True)

# Sayisal Değişkenlerin birbiri ile korelasyonu
correlation_matrix(df, num_cols)

# Target ile sayısal değişkenlerin ilişkisinin analizi
for col in num_cols:
    target_summary_with_num(df, "y", col, target_type="cat")

# Target ile kategorik değişkenlerin ilişkisinin analizi
for col in cat_cols:
    target_summary_with_cat(df, "y", col, target_type="cat")

################################################
# 2. Data Preprocessing & Feature Engineering
################################################
# Değişkenlerin tipi
df.info()

# Değişken isimlerinin büyütülmesi
df.columns = [col.upper() for col in df.columns]

# Yeni değişkenlerin türetilmesi
# df["NEW_AGE_GROUPED"] = pd.cut(df["AGE"], bins=[17, 25, 35, 50, 65, 100],
#                                labels=["Young", "Middle_Aged", "Adults", "Old", "Elder"])
df["NEW_PDAYS"] = df["PDAYS"].apply(lambda x: "No" if x == -1 else "Yes")
df["NEW_PREVIOUS"] = df["PREVIOUS"].apply(lambda x: "No" if x == 0 else "Yes")

# Eksik değerlerin incelenmesi
df.isnull().sum()

# Aykırı değerlerin incelenmesi
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)
for col in num_cols:
    print(col, check_outlier(df, col, q1=0.01, q3=0.99))

# Aykırı değerlerin tek değişkenli olarak yakalanması
for col in num_cols:
    grab_outliers(df, col, index=True, plot=True)

# Aykırı değerlerin baskılanması
for col in num_cols:
    replace_with_thresholds(df, col)

# Aykırı değerlerin çok değişkenli olarak yakalanması ve silinmesi
local_outliers = local_outlier_factor(df, num_cols, plot=False)
df = df.loc[~df.index.isin(local_outliers)]

# Bağımlı ve bağımsız değişkenlerin ayırılması
yes_df = df.loc[df["Y"] == "yes"].sample(5280)
no_df = df.loc[df["Y"] == "no"].sample(5280)
df = pd.concat([no_df, yes_df], axis=0)

Y = df["Y"]
X = df.drop("Y", axis=1)

# Encoding işlemlerinin yapılması
cat_cols, num_cols, cat_but_car = grab_col_names(X, cat_th=10, car_th=20)

X_encoded = one_hot_encoder(X, cat_cols, drop_first=True)
Y_encoded = pd.Series(LabelEncoder().fit_transform(Y))

X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y_encoded, test_size=0.20, random_state=42)

#######################
# 3. Base Models
#######################

base_models_clf(X_train, y_train, ["f1", "roc_auc", "accuracy"])

######################################################
# 4. Automated Hyperparameter Optimization
######################################################

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

best_models = hyperparameter_optimization_clf(X_train, y_train, classifiers=classifiers, cv=2,
                                              scoring=["f1", "recall", "precision", "roc_auc", "accuracy"])

model = best_models["RF"].fit(X_train, y_train)
plot_importance(model, X_train, len(X_train.columns))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#            0       0.89      0.83      0.86      1058
#            1       0.84      0.90      0.87      1054
#     accuracy                           0.86      2112
#    macro avg       0.86      0.86      0.86      2112
# weighted avg       0.86      0.86      0.86      2112
