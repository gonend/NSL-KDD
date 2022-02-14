import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import Preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from rotation_forest import RotationForestClassifier
import shap


if __name__ == '__main__':

    X_train, x_val, X_test, y_train, y_val, y_test = Preprocess.load_kdd_data()

    class_weights = Preprocess.calculating_class_weights(y_train)
    class_weights = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2], 3: class_weights[3],
                     4: class_weights[4]}

    ##TODO: play with the params.
    RF = RandomForestClassifier(class_weight=class_weights)
    GNB = GaussianNB()
    Adb = AdaBoostClassifier()
    RoF = RotationForestClassifier(class_weight=class_weights)

    print('-'*80)
    print(RF.__class__.__name__)
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    print(f"f1:{f1_score(y_test, y_pred,average='weighted')}")
    print(f"acc: {accuracy_score(y_test,y_pred)}")


    print('-' * 80)
    print(GNB.__class__.__name__)
    GNB.fit(X_train, y_train)
    y_pred = GNB.predict(X_test)
    print(f"f1:{f1_score(y_test, y_pred,average='weighted')}")
    print(f"acc: {accuracy_score(y_test,y_pred)}")


    print('-' * 80)
    print(Adb.__class__.__name__)
    Adb.fit(X_train, y_train)
    y_pred = Adb.predict(X_test)
    print(f"f1:{f1_score(y_test,y_pred,average='weighted')}")
    print(f"acc: {accuracy_score(y_test,y_pred)}")

    print('-' * 80)
    print(RoF.__class__.__name__)
    RoF.fit(X_train, y_train)
    y_pred = RoF.predict(X_test)
    print(f"f1:{f1_score(y_test, y_pred, average='weighted')}")
    print(f"acc: {accuracy_score(y_test, y_pred)}")

    """ generating shap values from XGBoost """
    # gb = define_fit_XGBoost(x_train_gb, y_train_gb, y_test, x_test, "XGBoost_" + dataset_name, data_path)
    RF_explainer = shap.TreeExplainer(RF)
    shap_values_gb = RF_explainer.shap_values(X_test)
    shap.summary_plot(shap_values_gb, X_test, plot_type="bar", max_display=20,
                      plot_size=(15, 15))

    """ sorted features importance names"""
    vals = np.abs(shap_values_gb).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    # feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    feature_importance.head()

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


