from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import Preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from rotation_forest import RotationForestClassifier

import shap

TARGET = 'attack_flag'

if __name__ == '__main__':

    df = Preprocess.load_kdd_data()

    y = df.attack_flag
    del df['attack_flag']
    X = df

    print(X.shape)
    print(y.shape)

    ##TODO: play with the params.
    RF = RandomForestClassifier()
    GNB = GaussianNB()
    Adb = AdaBoostClassifier()
    RoF = RotationForestClassifier()

    skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=True)
    models_accuracy = {}
    model_list = [RF,GNB,Adb,RoF]
    for model in model_list:
        cur_model = model.__class__.__name__
        models_accuracy[cur_model] = []
        for train_index,test_index in skf.split(X,y):
            X_train,X_test = X.iloc[train_index], X.iloc[test_index]
            y_train,y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            score = accuracy_score(pred,y_test)
            models_accuracy[cur_model].append(score)

        print('-' * 40 + cur_model + '-' * 40)
        print(f"acc: {np.array(models_accuracy[cur_model]).mean()}")




    # print('-'*80)
    # print(RF.__class__.__name__)
    # RF.fit(X_train,y_train)
    # y_pred = RF.predict(X_test)
    # print(f"f1:{f1_score(y_test, y_pred,average='weighted')}")
    # print(f"acc: {accuracy_score(y_test,y_pred)}")
    #
    #
    # print('-' * 80)
    # print(GNB.__class__.__name__)
    # GNB.fit(X_train,y_train)
    # y_pred = GNB.predict(X_test)
    # print(f"f1:{f1_score(y_test, y_pred,average='weighted')}")
    # print(f"acc: {accuracy_score(y_test,y_pred)}")
    #
    #
    # print('-' * 80)
    # print(Adb.__class__.__name__)
    # Adb.fit(X_train,y_train)
    # y_pred = Adb.predict(X_test)
    # print(f"f1:{f1_score(y_test,y_pred,average='weighted')}")
    # print(f"acc: {accuracy_score(y_test,y_pred)}")
    #
    # print('-' * 80)
    # print(RoF.__class__.__name__)
    # RoF.fit(X_train, y_train)
    # y_pred = RoF.predict(X_test)
    # print(f"f1:{f1_score(y_test, y_pred, average='weighted')}")
    # print(f"acc: {accuracy_score(y_test, y_pred)}")

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


