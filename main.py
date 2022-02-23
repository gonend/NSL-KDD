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
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay, recall_score,precision_score
from rotation_forest import RotationForestClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import SelectKBest

import shap

TARGET = 'attack_flag'

def run_models(flag=False,shap_flag=True,train=None,test=None):
    X_train, y_train = train.iloc[:, :-1], train[TARGET]
    X_test, y_test = test.iloc[:, :-1], test[TARGET]

    class_weights = Preprocess.calculating_class_weights(y_train)
    class_weights = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2], 3: class_weights[3],
                     4: class_weights[4]}

    if flag:
        print("Models Without Parameter Tuning")
        DT = DecisionTreeClassifier()
        RF = RandomForestClassifier(class_weight=class_weights, max_depth=3)  # , criterion='entropy', min_samples_split=20)
        RoF = RotationForestClassifier(class_weight=class_weights, max_depth=3)  # , criterion='entropy', min_samples_split=20)
        LR = LogisticRegression()
        xgb =XGBClassifier()
        Adb = AdaBoostClassifier()
        Knn = KNeighborsClassifier(n_neighbors=3)
        # RoF = RotationForestClassifier(class_weight=class_weights, min_samples_split=5) #, criterion='entropy')
        # RF = RandomForestClassifier(class_weight=class_weights, min_samples_split=5) #, criterion='entropy')

        models = [DT, RF, RoF, Adb, Knn] #, LR,xgb
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=True)
        for model in models:
            for train_inx,val_inx in skf.split(X_train,y_train):
                X_train, X_test = X_train.iloc[train_inx], X_train.iloc[val_inx]
                y_train, y_test = y_train.iloc[train_inx], y_train.iloc[val_inx]
                model.fit(X_train, y_train)

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
    run_models(flag=True,shap_flag=False,train=train,test=test)




    # # Create object that can calculate shap values
    # explainer = shap.TreeExplainer(RF)
    #
    # # Calculate Shap values
    # shap_values = explainer.shap_values(x_test)
    #
    # shap.initjs()
    # shap_plot = shap.force_plot(explainer.expected_value,
    #                             shap_values[-1:], features=x_test.iloc[-1:],
    #                             feature_names=x_test.columns[0:20],
    #                             matplotlib=True, show=False, plot_cmap=['#77dd77', '#f99191'])


    #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=True)
    # models_accuracy = {}
    # model_list = [RF, GNB, Adb, RoF]
    # for model in model_list:
    #     cur_model = model.__class__.__name__
    #     models_accuracy[cur_model] = []
    #     for train_index, test_index in skf.split(X, y):
    #         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #         model.fit(X_train, y_train)
    #         pred = model.predict(X_test)
    #         score = accuracy_score(pred, y_test)
    #         models_accuracy[cur_model].append(score)
    #
    #     print('-' * 40 + cur_model + '-' * 40)
    #     print(f"acc: {np.array(models_accuracy[cur_model]).min()}")#.mean()
