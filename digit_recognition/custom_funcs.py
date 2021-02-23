from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import itertools

def initial_sss(df, label, test_size, out_file):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=42)

    print(f'Spliting data: \n With shape of: {df.shape} \n Label being: {label} \n Output path: {out_file}')

    for train_index, test_index in sss.split(df, df[label]):
        train_df = df.loc[train_index]
        test_df = df.loc[test_index]

    print(f'Train shape: {train_df.shape}')
    print(f'Test shape: {test_df.shape}')

    train_df.to_csv(f'{out_file}/train_df.csv', index=False)
    test_df.drop(label, 1).to_csv(f'{out_file}/test_df.csv', index=False)
    test_df[label].to_csv(f'{out_file}/test_df_y_true.csv', index=False)

    print('Split successful')


def transform_pred_data(path):
    """ This function Transforms input data for prediction and ouputs
         them ready for evaluationg on model."""
    X_test = pd.read_csv(path)
    X_test  = X_test / 255
    X_test = X_test.values.reshape(-1, 28, 28,1)
    # plt.imshow(X_test[2]) for testing if data was reshaped properly
    print(X_test.shape)

    return X_test


def plot_missclassified_images(y_true, y_pred, X_test, n_images):
    print(y_true.shape)
    print(y_pred.shape)
    print(X_test.shape)
    #to add 1 dimension bcs shapes doesnt match with y_true
    y_pred = np.array(y_pred, ndmin=2).T

    errors_indexes = np.where(y_pred != y_true)[0]
    for _ in range(n_images):
        error_index = np.random.choice(errors_indexes)
        img = X_test[error_index]
        plt.imshow(img)
        plt.title(
            f'Predicted: {y_pred[error_index]} \n True: {y_true[error_index]}')
        plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()