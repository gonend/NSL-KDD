import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
import Preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    recall_score, precision_score
from rotation_forest import RotationForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest

import shap

TARGET = 'attack_flag'


def evaluate(model, X_train, y_train, X_test, y_test):
    print('-' * 40 + model.__class__.__name__ + '-' * 40)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"f1:{f1_score(y_test, y_pred, average='weighted')}")
    print(f"acc: {accuracy_score(y_test, y_pred)}")
    print(f"precision: {precision_score(y_test, y_pred, average='weighted')}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()


def run_models(grid_search=False, shap_flag=True, train=None, test=None):
    X_train, y_train = train.iloc[:, :-1], train[TARGET]
    X_test, y_test = test.iloc[:, :-1], test[TARGET]

    class_weights = Preprocess.calculating_class_weights(y_train)
    class_weights = {0: class_weights[0],
                     1: class_weights[1]}  # , 2: class_weights[2], 3: class_weights[3], 4: class_weights[4]

    DT = DecisionTreeClassifier()
    RF = RandomForestClassifier(class_weight=class_weights, max_depth=3)  # , criterion='entropy', min_samples_split=20)
    RoF = RotationForestClassifier(class_weight=class_weights,
                                   max_depth=3)  # , criterion='entropy', min_samples_split=20)
    LR = LogisticRegression()
    xgb = XGBClassifier()
    Adb = AdaBoostClassifier()
    Knn = KNeighborsClassifier(n_neighbors=3)
    # RoF = RotationForestClassifier(class_weight=class_weights, min_samples_split=5) #, criterion='entropy')
    # RF = RandomForestClassifier(class_weight=class_weights, min_samples_split=5) #, criterion='entropy')
    models = [DT, RF, RoF, Adb]

    if not grid_search:
        print("Models Without Parameter Tuning")
        # , LR,xgb, , Knn
        for model in models:
            evaluate(model, X_train, y_train, X_test, y_test)

    if grid_search:

        models_params = {}
        models_params["DecisionTreeClassifier"] = {'max_depth': [5, 10, 12, 15], 'min_samples_split': [2, 3, 5, 7]}
        models_params["RandomForestClassifier"] = {'n_estimators': [100, 200], 'max_depth': [5, 10, 15, 20, None],
                                                   'max_features': ['sqrt', None],
                                                   'min_samples_split': [2, 3], 'min_samples_leaf': [2, 5],
                                                   'class_weight': [None, 'balanced']}
        models_params["RotationForestClassifier"] = {'n_features_per_subset': [3, 5], 'max_depth': [5, 10, 15, None],
                                                     'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2],
                                                     'class_weight': [None, 'balanced']}
        models_params["AdaBoostClassifier"] = None
        for model in models:
            print(f"cur model {model.__class__.__name__}")
            dtc_gs = GridSearchCV(model, models_params[model.__class__.__name__], cv=5).fit(X_train, y_train)
            best_params = dtc_gs.best_params_
            print(best_params)
            if model.__class__.__name__ == DecisionTreeClassifier.__class__.__name__:
                model = DecisionTreeClassifier(max_depth=best_params['max_depth'],
                                               min_samples_split=best_params['min_samples_split'])
            elif model.__class__.__name__ == RandomForestClassifier.__class__.__name__:
                model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                               max_depth=best_params['max_depth'],
                                               min_samples_split=best_params['min_samples_split'],
                                               min_samples_leaf=best_params['min_samples_leaf'],
                                               max_features=best_params['max_features'])
            elif model.__class__.__name__ == RandomForestClassifier.__class__.__name__:
                model = RotationForestClassifier(n_features_per_subset=best_params['n_features_per_subset'],
                                                 max_depth=best_params['max_depth'],
                                                 min_samples_split=best_params['min_samples_split'],
                                                 min_samples_leaf=best_params['min_samples_leaf'])

            elif model.__class__.__name__ == AdaBoostClassifier:
                break
            evaluate(model, X_train, y_train, X_test, y_test)

        ##TODO: Grid Search to optimize hyperParamters,
        ##TODO: Delete irelvant Feature

    if shap_flag:
        """ generating shap values from XGBoost """
        # gb = define_fit_XGBoost(x_train_gb, y_train_gb, y_test, x_test, "XGBoost_" + dataset_name, data_path)
        RF_explainer = shap.TreeExplainer(RF)
        shap_values_gb = RF_explainer.shap_values(X_test)
        shap.summary_plot(shap_values_gb, X_test, plot_type="bar", max_display=20,
                          plot_size=(15, 15))

        """ sorted features importance names"""
        vals = np.abs(shap_values_gb).mean(0)
        feature_importance = pd.DataFrame(list(zip(X_train.columns, vals)),
                                          columns=['col_name', 'feature_importance_vals'])
        # feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        feature_importance.head()


if __name__ == '__main__':
    # X_train, x_val, X_test, y_train, y_val, y_test = Preprocess.load_kdd_data()
    train, test = Preprocess.read_and_preprocess_kdd()
    run_models(grid_search=False, shap_flag=False, train=train, test=test)
    print("-" * 40 + "Runing models with improvments" + "-" * 40)
    print(train.shape), print(test.shape)
    train, test = Preprocess.read_and_preprocess_kdd(plots=True)
    run_models(grid_search=True, shap_flag=False, train=train, test=test)

#    for train_inx, val_inx in skf.split(X_train, y_train):
#     X_train1, X_val = X_train.iloc[train_inx], X_train.iloc[val_inx]
#     y_train1, y_val = y_train.iloc[train_inx], y_train.iloc[val_inx]
#     model.fit(X_train1, y_train1)
#     y_pred = model.predict(X_val)
#     score = cohen_kappa_score(y_pred, y_val)
#     models_accuracy[model].append(score)
# print('-' * 40 + model.__class__.__name__ + '-' * 40)
# print(f"validation acc: {np.array(models_accuracy[model]).mean()}")#.mean()
# y_pred = model.predict(X_test)
# print(f"test acc: {cohen_kappa_score(y_pred,y_test)}")
