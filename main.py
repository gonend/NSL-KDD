import os

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
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    recall_score, precision_score
from rotation_forest import RotationForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest
from sklearn.datasets import make_classification
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

import shap

TARGET = 'attack_flag'


def evaluate(model, X_train, y_train, X_test, y_test, fit):
    print('-' * 40 + model.__class__.__name__ + '-' * 40)
    if fit:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"f1:{f1_score(y_test, y_pred, average='weighted')}")
    print(f"acc: {accuracy_score(y_test, y_pred)}")
    print(f"precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"recall: {recall_score(y_test, y_pred)}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    disp.ax_.set_title(model.__class__.__name__, fontsize=15)
    plt.show()


def run_models(tuning=False, strat=False, shap_flag=True, train=None, test=None):
    X_train, y_train = train.iloc[:, :-1], train[TARGET]
    X_test, y_test = test.iloc[:, :-1], test[TARGET]

    # # embedding
    # extract = Preprocess.embedding(X_train)
    # X_train = extract.predict(X_train)
    # X_test = extract.predict(X_test)

    class_weights = Preprocess.calculating_class_weights(y_train)
    class_weights = {0: class_weights[0],
                     1: class_weights[1]}  # , 2: class_weights[2], 3: class_weights[3], 4: class_weights[4]

    DT = DecisionTreeClassifier()
    # RF = RandomForestClassifier(class_weight=class_weights, max_depth=5)  # , criterion='entropy', min_samples_split=20)
    RF = RandomForestClassifier()  # max_depth=5 , criterion='entropy', min_samples_split=20)
    # RoF = RotationForestClassifier(class_weight=class_weights, max_depth=3)  # , criterion='entropy', min_samples_split=20)
    RoF = RotationForestClassifier()  # max_depth=3 , criterion='entropy', min_samples_split=20)
    LR = LogisticRegression()
    Adb = AdaBoostClassifier()
    xgb = XGBClassifier()
    # RoF = RotationForestClassifier(class_weight=class_weights, min_samples_split=5) #, criterion='entropy')
    # RF = RandomForestClassifier(class_weight=class_weights, min_samples_split=5) #, criterion='entropy')

    # DT = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=3)
    # RF = RandomForestClassifier(criterion='entropy', max_depth=3, min_samples_split=3,
    #                        n_estimators=20)
    # RoF = RotationForestClassifier(criterion='entropy', max_depth=15, min_samples_split=5,
    #                          n_estimators=20)
    # Adb = AdaBoostClassifier(learning_rate=1, n_estimators=20)

    models = [DT, RF, RoF, Adb, xgb]  # , xgb]

    if not tuning:
        print("-" * 40 + "Runing models without improvments" + "-" * 40)
        for model in models:
            evaluate(model, X_train, y_train, X_test, y_test, fit=True)

    if tuning:
        print("-" * 40 + "Runing models with improvments" + "-" * 40)

        # DT = DecisionTreeClassifier(max_depth=5,criterion='entropy', min_samples_split=20) #
        # RF = RandomForestClassifier(n_estimators=150,max_depth=7, criterion='gini', min_samples_split=15) #, max_features='sqrt'
        # RoF = RotationForestClassifier(n_estimators=50, max_depth=5, criterion='gini', min_samples_split=20)  # max_depth=3 , criterion='entropy', min_samples_split=20)
        # Adb = AdaBoostClassifier(n_estimators=100,learning_rate=1)

        # DT = DecisionTreeClassifier(max_depth=12, criterion='entropy', min_samples_split=20)  #
        # RF = RandomForestClassifier(n_estimators=120, max_depth=8, criterion='gini', min_samples_split=15)  #
        # RoF = RotationForestClassifier(n_estimators=50, max_depth=5, criterion='gini',
        #                                min_samples_split=20)  # max_depth=3 , criterion='entropy', min_samples_split=20)
        # Adb = AdaBoostClassifier(n_estimators=100, learning_rate=1)



        # # grid search
        # DT = DecisionTreeClassifier()  #
        # RF = RandomForestClassifier()  #
        # RoF = RotationForestClassifier()  # max_depth=3 , criterion='entropy', min_samples_split=20)
        # Adb = AdaBoostClassifier()
        # xgb = XGBClassifier()
        #
        #
        # models_params = dict()
        # models_params[DT.__str__()] = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'min_samples_split': [2, 3, 5, 10, 20],
        #                      'criterion': ['entropy', 'gini']}
        # models_params[RF.__str__()] = {'n_estimators': [10, 20, 50, 100, 120], 'max_depth': [3, 5, 8, 9, 10, 15, None],
        #                      'criterion': ['entropy', 'gini'], 'min_samples_split': [2, 3, 5, 10, 20]}
        # models_params[RoF.__str__()] = {'n_estimators': [10, 20, 50, 100, 120], 'max_depth': [3, 5, 8, 9, 10, 15, None],
        #                       'criterion': ['entropy', 'gini'], 'min_samples_split': [2, 3, 5, 10, 20]}
        # models_params[Adb.__str__()] = {'n_estimators': [20, 50, 100, 120], 'learning_rate': [1.0, 2.0, 3.0, 4.0, 5.0]}
        # models_params[xgb.__str__()] = {'n_estimators': [20, 50, 100, 120], 'use_label_encoder': [True, False]}

        # DT = DecisionTreeClassifier(criterion='gini', max_depth=3)
        # RF = RandomForestClassifier(criterion='entropy', max_depth=3, n_estimators=20)
        # RoF = RotationForestClassifier(criterion='entropy', max_depth=3)
        # Adb = AdaBoostClassifier(n_estimators=20)
        # xgb = XGBClassifier()

        DT = DecisionTreeClassifier(criterion='entropy', max_depth=10)
        RF = RandomForestClassifier(criterion='entropy', max_depth=5, n_estimators=10)
        RoF = RotationForestClassifier(criterion='entropy', max_depth=3, n_estimators=50)
        Adb = AdaBoostClassifier(n_estimators=20)
        xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                      gamma=0, gpu_id=-1, importance_type=None,
                      interaction_constraints='', learning_rate=0.300000012,
                      max_delta_step=0, max_depth=6, min_child_weight=1,
                      monotone_constraints='()', n_estimators=20, n_jobs=16,
                      num_parallel_tree=1, predictor='auto', random_state=0,
                      reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                      tree_method='exact', validate_parameters=1, verbosity=None)



        models = [DT, RF, RoF, Adb, xgb]
        # models = [xgb]  # , xgb]
        models_accuracy = {}
        skf = StratifiedKFold(n_splits=4)
        for model in models:
            if strat:
                models_accuracy[model] = []
                for train_inx, val_inx in skf.split(X_train, y_train):
                    X_train1, X_val = X_train.iloc[train_inx], X_train.iloc[val_inx]
                    y_train1, y_val = y_train.iloc[train_inx], y_train.iloc[val_inx]
                    model.fit(X_train1, y_train1)
                    y_pred = model.predict(X_val)
                    score = accuracy_score(y_pred, y_val)
                    models_accuracy[model].append(score)
                # print('-' * 40 + model.__class__.__name__ + '-' * 40)

                evaluate(model, X_train, y_train, X_test, y_test, fit=False)
                print(f"validation acc: {np.array(models_accuracy[model]).mean()}")  # .mean()
            else:
                pass
                # param_grid = models_params[model.__str__()]
                # base_estimator = model
                # X, y = X_train, y_train
                # sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                #                          factor=2,
                #                          max_resources=30).fit(X, y)
                # sh.best_estimator_
                # print(sh.best_estimator_)



                # evaluate(model, X_train, y_train, X_test, y_test, fit=True)

            # y_pred = model.predict(X_test)
            # print(f"test acc: {accuracy_score(y_pred,y_test)}")

        # RF = RandomForestClassifier(class_weight=class_weights, max_depth=5)  # , criterion='entropy', min_samples_split=20)
        # RoF = RotationForestClassifier(class_weight=class_weights, max_depth=3)  # , criterion='entropy', min_samples_split=20)
        # LR = LogisticRegression()

        # models_params = {}
        # models_params["DecisionTreeClassifier"] = {'max_depth': [5, 10, 12, 15, None], 'min_samples_split': [2, 3, 5, 7]}
        # models_params["RandomForestClassifier"] = {'n_estimators': [50, 100], 'max_depth': [5, 10, 15, None],
        #                                           'max_features': ['sqrt', None],
        #                                           'min_samples_split': [2, 3], 'min_samples_leaf': [2, 5],
        #                                           'class_weight': [None, 'balanced']}
        # models_params["RotationForestClassifier"] = {'n_features_per_subset': [3, 5], 'max_depth': [5, 10, 15, None],
        #                                             'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2],
        #                                             'class_weight': [None, 'balanced']}
        # models_params["AdaBoostClassifier"] = None

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
    if (os.path.isfile("data/kdd_after_preprocess_test.csv") == False):
        train, test = Preprocess.read_and_preprocess_kdd(flag=True)
    else:
        train = pd.read_csv("data/kdd_after_preprocess_train.csv")
        test = pd.read_csv("data/kdd_after_preprocess_test.csv")
        train = train.drop("Unnamed: 0", axis=1)
        test = test.drop("Unnamed: 0", axis=1)
    # run_models(tuning=False, shap_flag=False, train=train, test=test)

    # print("-" * 40 + "Not Using StratifiedKFold" + "-" * 40)
    # run_models(tuning=True, strat=False, shap_flag=False, train=train, test=test)

    print("-" * 40 + "Using StratifiedKFold" + "-" * 40)
    run_models(tuning=True, strat=True, shap_flag=True, train=train, test=test)

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


# print(f"cur model {model.__class__.__name__}")
# dtc_gs = GridSearchCV(model, models_params[model.__class__.__name__], cv=5).fit(X_train, y_train)
# best_params = dtc_gs.best_params_
# print(best_params)
# if model.__class__.__name__ == DecisionTreeClassifier.__class__.__name__:
#     model = DecisionTreeClassifier(max_depth=best_params['max_depth'],
#                                    min_samples_split=best_params['min_samples_split'])
# elif model.__class__.__name__ == RandomForestClassifier.__class__.__name__:
#     model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
#                                    max_depth=best_params['max_depth'],
#                                    min_samples_split=best_params['min_samples_split'],
#                                    min_samples_leaf=best_params['min_samples_leaf'],
#                                    max_features=best_params['max_features'])
# elif model.__class__.__name__ == RandomForestClassifier.__class__.__name__:
#     model = RotationForestClassifier(n_features_per_subset=best_params['n_features_per_subset'],
#                                      max_depth=best_params['max_depth'],
#                                      min_samples_split=best_params['min_samples_split'],
#                                      min_samples_leaf=best_params['min_samples_leaf'])
#
# elif model.__class__.__name__ == AdaBoostClassifier:
#     break
