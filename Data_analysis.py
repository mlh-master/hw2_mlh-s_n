import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def clean_data(df):
    """

    :param df: Pandas series of df
    :return: Dataframe of the cleaned features called clean
    """

    c_ctg = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    clean = {}

    for column_name in c_ctg.columns:
        index_hist = c_ctg.loc[:, column_name].dropna()

        def rand_sampling(x, var_hist):
            if np.isnan(x):
                rand_idx = np.random.choice(var_hist.index)
                x = var_hist[rand_idx]
            return x

        clean[column_name] = c_ctg[[column_name]].applymap(lambda x: rand_sampling(x, index_hist))[column_name]

    return pd.DataFrame(clean)

def standart(CTG_features):
    """

    :param CTG_features: Pandas series of CTG features
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """

    nsd_res = (CTG_features - CTG_features.min())/(CTG_features.max()-CTG_features.min())

    return pd.DataFrame(nsd_res)

def Comparison_test_train(X_train, X_test):
    """

    :param X_train: Pandas series of split df - training set
    :param X_test: Pandas series of split df - testing set
    :return: Dataframe of comparison between train&test called X_comparison
    """

    X_comparison = pd.DataFrame(columns=['train%','test%','Delta%'])
    X_comparison.index.name = 'Positive feature'

    for column_name in X_train.columns:
        if column_name != 'Age':
            X_train_proba= int(X_train[column_name].mean()*100)
            X_test_proba = int(X_test[column_name].mean()*100)
            X_comparison.loc[column_name] = [X_train_proba,X_test_proba,X_train_proba-X_test_proba]

    return X_comparison

def Feature_vs_Label(df, Diagnosis):
    """

    :param df: Pandas series of df
    :param Diagnosis: Pandas series of Diagnosis labels
    :return: None
    """


    i = 1
    width = 0.35  # the width of the bars

    fig, axs = plt.subplots(4, 4, figsize=(20, 15), facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()
    fig = plt.figure()

    for columns_name in df:
        if columns_name == 'Diagnosis':
            continue
        labels = df[columns_name].unique()
        col_1 = []
        col_2 = []
        for label in labels:
            col_1.append(len(df[(df[Diagnosis] == 'Positive') & (df[columns_name] == label)]))
            col_2.append(len(df[(df[Diagnosis] == 'Negative') & (df[columns_name] == label)]))

        x = np.arange(len(labels))  # the label locations

        rects1 = axs[i-1].bar(x - width / 2, col_1, width, label='Positive')
        rects2 = axs[i-1].bar(x + width / 2, col_2, width, label='Negative')

        # Add some text for labels, title and custom x-axis tick labels, etc.


        axs[i-1].set_ylabel('Count')
        axs[i-1].set_title(str(columns_name))
        axs[i-1].set_xticks(x)
        axs[i-1].set_xticklabels(labels)
        axs[i-1].legend()
        plt.tight_layout()


        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                axs[i-1].annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        i += 1
    plt.show()
    return None

def plt_2d_pca(X_pca,y,title):
    """
        :param X_pca: Pandas series of reduced dimenshionaly data
        :param Diagnosis: Pandas series of Diagnosis labels

    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='b')
    ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='r')

    ax.legend(('Negative','Positive'))
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel('$U_1$')
    ax.set_ylabel('$U_2$')
    ax.set_title(title)
    plt.show()