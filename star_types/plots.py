from itertools import cycle
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


def confusion_matrix_plot(y_test, predictions, classes):
    cm = confusion_matrix(y_test, predictions)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, cmap="YlGnBu", ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)


def roc_auc_multiclass(X_test, y_test, model_type=None, model=None):

    if model_type in ['logistic regression', 'random forest', 'kneighbors']:
        predictions = model.predict_proba(X_test)
    elif model_type == 'support vector machine':
        predictions = model.decision_function(X_test)
    n_classes = y_test.shape[1] 

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")

