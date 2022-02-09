from sklearn.metrics import f1_score

import Preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb



if __name__ == '__main__':
    x_train, x_val, x_attack, x_test, y_train, y_attack, y_val, y_test = Preprocess.load_kdd_data()
    ##TODO: play with the params.
    RF = RandomForestClassifier()
    GNB = GaussianNB()
    Adb = AdaBoostClassifier()

    print('-'*80)
    RF.fit(x_train,y_train)
    y_pred = RF.predict(x_test)
    print(f"f1:{f1_score(y_test, y_pred)}")
    print(f"acc: {accuracy_score(y_test,y_pred)}")


    print('-' * 80)
    GNB.fit(x_train,y_train)
    y_pred = GNB.predict(x_test)
    print(f"f1:{f1_score(y_test, y_pred)}")
    print(f"acc: {accuracy_score(y_test,y_pred)}")


    print('-' * 80)
    Adb.fit(x_train,y_train)
    y_pred = Adb.predict(x_test)
    print(f"f1:{f1_score(y_test,y_pred)}")
    print(f"acc: {accuracy_score(y_test,y_pred)}")

    #testing




