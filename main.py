from sklearn.metrics import f1_score

import Preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from rotation_forest import RotationForestClassifier
import shap


if __name__ == '__main__':

    X_train, x_val, X_attack, X_test, y_train, y_attack, y_val, y_test = Preprocess.load_kdd_data()

    ##TODO: play with the params.
    RF = RandomForestClassifier()
    GNB = GaussianNB()
    Adb = AdaBoostClassifier()
    RoF = RotationForestClassifier()

    print('-'*80)
    print(RF.__class__.__name__)
    RF.fit(X_train,y_train)
    y_pred = RF.predict(X_test)
    print(f"f1:{f1_score(y_test, y_pred,average='weighted')}")
    print(f"acc: {accuracy_score(y_test,y_pred)}")


    print('-' * 80)
    print(GNB.__class__.__name__)
    GNB.fit(X_train,y_train)
    y_pred = GNB.predict(X_test)
    print(f"f1:{f1_score(y_test, y_pred,average='weighted')}")
    print(f"acc: {accuracy_score(y_test,y_pred)}")


    print('-' * 80)
    print(Adb.__class__.__name__)
    Adb.fit(X_train,y_train)
    y_pred = Adb.predict(X_test)
    print(f"f1:{f1_score(y_test,y_pred,average='weighted')}")
    print(f"acc: {accuracy_score(y_test,y_pred)}")

    print('-' * 80)
    print(RoF.__class__.__name__)
    RoF.fit(X_train, y_train)
    y_pred = RoF.predict(X_test)
    print(f"f1:{f1_score(y_test, y_pred, average='weighted')}")
    print(f"acc: {accuracy_score(y_test, y_pred)}")

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


