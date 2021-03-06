import keras as keras
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
import itertools
# import joblib
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# from Comparison_Detection import RANDOM_SEED
# from scipy.stats import zscore
import seaborn as sb
import matplotlib.pyplot as mp
from scipy import stats
#from keras.layers import Dense
#from keras.callbacks import EarlyStopping
#from keras import Model

TARGET = 'attack_flag'

n_adv = 5
THRESHOLD = 0.5

"""
discretizes the target class into 5.
0: normal (no attack)
1: ddos\dos attack
2: probebility attack
3: privilege attack
4: access attack
"""


def target_bins(attack):
    # categorize the attacks

    dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop',
                   'udpstorm', 'worm']
    probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',
                      'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xclock', 'xsnoop']

    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 1
    elif attack in probe_attacks:
        # probe_attacks mapt to 2
        attack_type = 2
    elif attack in privilege_attacks:
        # privilege escalation attacks map to 3
        attack_type = 3
    elif attack in access_attacks:
        # remote access attacks map to 4
        attack_type = 4
    else:
        # normal maps to 0
        attack_type = 0

    return attack_type


def read_and_preprocess_kdd(flag: bool = False):
    # file_path_20_percent = 'data/KDDTrain+_20Percent.txt'
    file_path_full_training_set = 'data/KDDTrain+.txt'
    file_path_test = 'data/KDDTest+.txt'


    df = pd.read_csv(file_path_full_training_set)
    # df = pd.read_csv(file_path_full_training_set)
    test_df = pd.read_csv(file_path_test)

    columns = (['duration'
        , 'protocol_type'
        , 'service'
        , 'flag'
        , 'src_bytes'
        , 'dst_bytes'
        , 'land'
        , 'wrong_fragment'
        , 'urgent'
        , 'hot'
        , 'num_failed_logins'
        , 'logged_in'
        , 'num_compromised'
        , 'root_shell'
        , 'su_attempted'
        , 'num_root'
        , 'num_file_creations'
        , 'num_shells'
        , 'num_access_files'
        , 'num_outbound_cmds'
        , 'is_host_login'
        , 'is_guest_login'
        , 'count'
        , 'srv_count'
        , 'serror_rate'
        , 'srv_serror_rate'
        , 'rerror_rate'
        , 'srv_rerror_rate'
        , 'same_srv_rate'
        , 'diff_srv_rate'
        , 'srv_diff_host_rate'
        , 'dst_host_count'
        , 'dst_host_srv_count'
        , 'dst_host_same_srv_rate'
        , 'dst_host_diff_srv_rate'
        , 'dst_host_same_src_port_rate'
        , 'dst_host_srv_diff_host_rate'
        , 'dst_host_serror_rate'
        , 'dst_host_srv_serror_rate'
        , 'dst_host_rerror_rate'
        , 'dst_host_srv_rerror_rate'
        , 'attack'
        , 'level'])

    df.columns = columns
    test_df.columns = columns

    print(df.info())

    # df = pd.read_csv("data/kdd_after_preprocess_train.csv")

    # df = df.drop("Unnamed: 0", axis=1)

    # test_df = pd.read_csv("data/kdd_after_preprocess_test.csv")

    # test_df = test_df.drop("Unnamed: 0", axis=1)

    # # set target variabels into new classes-> multi class classification
    # is_attack = df.attack.apply(target_bins)
    #
    # test_attack = test_df.attack.apply(target_bins)
    # binary classification

    is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
    test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)
    df['attack_flag'] = is_attack
    test_df['attack_flag'] = test_attack

    # delete data leakage
    del df["attack"]
    del test_df["attack"]

        ## encode train_data
    le = LabelEncoder()
    cols_to_label_encode = ['protocol_type', 'service', 'flag']
    for col in cols_to_label_encode:
        df[col] = le.fit_transform(df[col])
        test_df[col] = le.fit_transform(test_df[col])

        # df['protocol_type'] = le.fit_transform(df['protocol_type'])
        # test_df['protocol_type'] = le.transform(test_df['protocol_type'])
        # df['service'] = le.fit_transform(df['service'])
        # test_df['service'] = le.transform(test_df['service'])
        # df['flag'] = le.fit_transform(df['flag'])
        # test_df['flag'] = le.transform(test_df['flag'])

    # SAME VAL
    for col in df.columns:
        # feature_plot(df, col, 'Train')
        to_del = imbalance_features(df, col)
        if to_del:
            #print(col)
            del df[col]
            del test_df[col]
        # feature_plot(test_df,TARGET,'Test')

    df.corr(method='pearson')
    dataplot = sb.heatmap(df.corr())
    mp.show()

    corrdict = {col: [] for col in df.columns}
    for i in range(len(df.columns)):
        # for col2 in df.columns:
        if i < len(df.columns) - 1:
            if df[df.columns[i]].corr(df[df.columns[i + 1]]) > 0.85:
                # if col1 not in corrdict.values():
                # if col2 not in corrdict[col1] and col1 not in corrdict[col2]:
                # print(df[df.columns[i]].name)
                corrdict[df[df.columns[i]].name] = corrdict[df[df.columns[i]].name] + [df[df.columns[i + 1]].name]
                # if col2 not in corrdict.values():
                # if col1 not in corrdict[col2]:
                #     corrdict[col2] = corrdict[col2] + [col1]
    res = {col: corrdict[col] for col in df.columns if corrdict[col] != []}
    # print(res)
    s = []
    for c in res.keys():
        s.append(set([c, *res[c]]))
        # print((c, *res[c]))
    for i in s:
        print(i)

    # # delete for correlation
    # # del df['num_compromised']
    # # del test_df['num_compromised']
    # # del df['num_root']
    # # del test_df['num_root']
    del df['dst_host_srv_serror_rate']
    del test_df['dst_host_srv_serror_rate']
    del df['srv_serror_rate']
    del test_df['srv_serror_rate']
    # del df['dst_host_serror_rate']
    # del test_df['dst_host_serror_rate']
    del df['dst_host_srv_rerror_rate']
    del test_df['dst_host_srv_rerror_rate']
    del df['srv_rerror_rate']
    del test_df['srv_rerror_rate']
    del df['dst_host_same_srv_rate']
    del test_df['dst_host_same_srv_rate']


    # delete for same-diff
    del df['srv_count']
    del test_df['srv_count']
    del df['diff_srv_rate']
    del test_df['diff_srv_rate']
    # del df['dst_host_same_srv_rate']
    # del test_df['dst_host_same_srv_rate']
    del df['dst_host_srv_diff_host_rate']
    del test_df['dst_host_srv_diff_host_rate']

        # print(len(set(df['level'])))
        # print("---")
        # print(set(df['level']))
        # print("---")
        # print(df['level'])
        # print("---")

        # c = 0
        # for i in range(len(df["dst_host_same_src_port_rate"])):
        #     if df['dst_host_same_src_port_rate'][i] != df['dst_host_srv_diff_host_rate'][i]:
        #         c += 1
        # print(c)

        # c0 = 0
        # c1 = 0
        # c2 = 0
        # for i in df["logged_in"]:
        #     if i == 0:
        #         c0 += 1
        #     elif i == 1:
        #         c1 += 1
        #     else:
        #         c2 += 1
        # print(c0)
        # print(c1)
        # print(c2)


        # cols_to_zscore = [col for col in df.columns if col not in cols_to_label_encode]
        ## 'src_bytes', 'dst_bytes', 'count', 'dst_host_count', 'dst_host_srv_count', 'level' - ??????????
        ## ???????? ?????????? srv_count ???? ???? ?????? ?? count
        ## ???????? ?????????? diff_srv_rate ???? ???????? ?? same_srv_rate
        ## ???????? ?????????? dst_host_same_srv_rate ???? ???????? ?? dst_host_same_srv_rate
        ## ???????? ?????????? dst_host_srv_diff_host_rate ???? ???????? ?? dst_host_same_src_port_rate
        ## 'serror_rate' between 0-1
        ## 'rerror_rate' between 0-1
        ## 'same_srv_rate' between 0-1
        ## 'srv_diff_host_rate' between 0-1
        ## 'dst_host_same_srv_rate' between 0-1
        ## 'dst_host_same_src_port_rate' between 0-1
        ## 'dst_host_rerror_rate' between 0-1

    cols_to_min_max_normalization = ['src_bytes', 'dst_bytes', 'count', 'dst_host_count', 'dst_host_srv_count', 'level']
    # 'protocol_type', 'service', 'flag'
    min_max_scaler = MinMaxScaler()
    for col in cols_to_min_max_normalization:
        df[[col]] = min_max_scaler.fit_transform(df[[col]])
        test_df[[col]] = min_max_scaler.fit_transform(test_df[[col]])
            # stats.zscore(df[col])
            # stats.zscore(test_df[col])

        # for col in cols_to_zscore:
        #     stats.zscore(df[col])
        #     stats.zscore(test_df[col])

        # # detele from shap
        # del df['duration']
        # del test_df['duration']
        # del df['land']
        # del test_df['land']
        # del df['wrong_fragment']
        # del test_df['wrong_fragment']
        # del df['urgent']
        # del test_df['urgent']
        # del df['hot']
        # del test_df['hot']
        # del df['num_failed_logins']
        # del test_df['num_failed_logins']
        # del df['num_compromised']
        # del test_df['num_compromised']
        # del df['root_shell']
        # del test_df['root_shell']
        # del df['su_attempted']
        # del test_df['su_attempted']
        # del df['num_root']
        # del test_df['num_root']
        # del df['num_file_creations']
        # del test_df['num_file_creations']
        # del df['num_shells']
        # del test_df['num_shells']
        # del df['num_access_files']
        # del test_df['num_access_files']
        # del df['num_outbound_cmds']
        # del test_df['num_outbound_cmds']
        # del df['is_host_login']
        # del test_df['is_host_login']
        # del df['is_guest_login']
        # del test_df['is_guest_login']
        # del df['srv_count']
        # del test_df['srv_count']
        # del df['srv_serror_rate']
        # del test_df['srv_serror_rate']
        # del df['srv_diff_host_rate']
        # del test_df['srv_diff_host_rate']
        # del df['dst_host_count']
        # del test_df['dst_host_count']
        # del df['dst_host_srv_serror_rate']
        # del test_df['dst_host_srv_serror_rate']
        # del df['dst_host_rerror_rate']
        # del test_df['dst_host_rerror_rate']
        # # del df['attack']
        # # del test_df['attack']

        # df = df.astype('float32')
        # test_df = test_df.astype('float32')

        # print(df.info())

        # df = df.astype('float32')
        # test_df = test_df.astype(('float32'))
    if flag:
        # print feature distribution and if it greater then 0.95
        for col in df.columns:
            # feature_plot(df, col, 'Train')
            to_del = imbalance_features(df, col)
            if to_del:
                del df[col]
                del test_df[col]
            # feature_plot(test_df,TARGET,'Test')

    # df.corr(method='pearson')
    # dataplot = sb.heatmap(df.corr())
    # corrdict = {col: [] for col in df.columns}
    # for col1 in df.columns:
    #     for col2 in df.columns:
    #         if col1 != col2 and df[col1].corr(df[col2]) > 0.95:
    #             # if col1 not in corrdict.values():
    #             if col2 not in corrdict[col1] and col1 not in corrdict[col2]:
    #                 corrdict[col1] = corrdict[col1] + [col2]
    #             # if col2 not in corrdict.values():
    #             # if col1 not in corrdict[col2]:
    #             #     corrdict[col2] = corrdict[col2] + [col1]
    # res = {col: corrdict[col] for col in df.columns if corrdict[col] != []}
    # s = []
    # for c in res.keys():
    #     s.append(set([c, *res[c]]))
    #     # print((c, *res[c]))
    # for i in s:
    #     print(i)
    #             # print(col1, col2, df[col1].corr(df[col2]))
    # mp.show()

    # print(df['num_compromised'])
    # print(df['num_root'])
    # print('-'*20)
    # # print(df['dst_host_srv_serror_rate'])
    # # print(df['srv_serror_rate'])
    # print(df['serror_rate'])
    # # print(df['dst_host_serror_rate'])
    # print('-'*20)
    # print(df['rerror_rate'])
    # print(df['dst_host_srv_rerror_rate'])
    # print(df['srv_rerror_rate'])
    # print('-'*20)

    df.to_csv("data/kdd_after_preprocess_train.csv")
    test_df.to_csv("data/kdd_after_preprocess_test.csv")

    print(df.info())

    return df, test_df


def print_class_freq(df):
    for i in range(0, 5):
        print(f"Class {i} frequency: ")
        print(df[df[TARGET] == i].shape[0])


def balanced_train_data(df_train):
    """
    sampeling to generate balanced train dataset
    """
    hate_train_df = df_train[df_train.pred == 1]
    print("hate_train_df shape:", hate_train_df.shape)
    df2 = df_train[df_train[TARGET] == 0].sample(len(hate_train_df) * 5, axis=0, random_state=2)
    df_train = pd.concat([hate_train_df, df2], axis=0, sort=False)
    print("df shape: ", df_train.shape)
    return df_train


def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    # number_dim = np.shape(y_true)[1]
    # weights = np.empty([number_dim, 2])
    # weights = compute_class_weight(class_weight='balanced', classes=[0., 1., 2., 3., 4.], y=y_true)
    weights = compute_class_weight(class_weight='balanced', classes=[0., 1.], y=y_true)
    # weights =  dict(zip(np.unique(train_classes), class_weights))
    return weights


def feature_plot(df, feature, title):
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.hist(df[feature], density=False, bins=20, ec='black')
    ax.set_title(f'{title}: {feature} Distribution', fontsize=15)

    fig.suptitle(title, fontsize=20)
    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    plt.show()
    fig.savefig(f'plots/{feature}.png')


def imbalance_features(df, feature):
    vc = df[feature].value_counts()
    m = max(vc)
    s = sum(vc)
    if m / s > 0.95:
        return True
    return False


# def embedding(x_train):
#     model = keras.Sequential()
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(128, input_dim=64, activation='relu'))
#     model.add(Dense(16, input_dim=128, activation='relu', name="layer4"))
#     model.add(Dense(128, input_dim=64, activation='relu'))
#     model.add(Dense(x_train.shape[1], activation='sigmoid'))
#     # Compile model
#     model.compile(optimizer='adam', loss='mean_absolute_error')
#
#     # class_weights = calculating_class_weights(y_train)
#     # class_weights = {0: class_weights[0], 1: class_weights[1]}
#     early_stopping_monitor = EarlyStopping(monitor='loss', restore_best_weights=True, patience=7)
#
#     # if (os.path.isfile(data_path + "/models/EMBD_" + dataset_name + str(RANDOM_SEED) + ".h5") == False):
#     model.fit(x_train, x_train,
#               verbose=1,
#               epochs=80,
#               batch_size=10,
#               # validation_data=(x_val, x_val),
#               callbacks=[early_stopping_monitor])
#
#     # model.save(data_path + "/models/EMBD_" + dataset_name + str(RANDOM_SEED) + ".h5")  # creates a HDF5 file
#     # else:
#     #     model = load_model(data_path + "/models/EMBD_" + dataset_name + str(RANDOM_SEED) + ".h5")
#
#     extract = Model(inputs=model.inputs, outputs=model.layers[-3].output)
#     return extract
