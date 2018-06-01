import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

#plot ROC from models
def printRoc(y_test,y_test_cnn,df):

    labels = [
       {'name': "RandomForestClassifier"},
        {'name': "LinearSVC"},
        {'name': 'MultinomialNB'},
        {'name': 'Logistic Regresion'},
    ]

    plt.figure(figsize=(8, 7))

    #Calculate ROC for models on labels
    for l in labels:
        fpr, tpr, threshold = roc_curve(y_test,df[l['name']])
        roc_auc= auc(fpr, tpr)
        plt.plot(fpr, tpr, label=l['name'] + ' (area = %0.3f)' % roc_auc, linewidth=2)

    #Calculate ROC FOR CNN
    fpr, tpr, threshold = roc_curve(y_test_cnn, df["CNN"].values)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="CNN" + ' (area = %0.3f)' % roc_auc, linewidth=2)

    #Plot the ROC for all models
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Receiver operating characteristic: is positive', fontsize=18)
    plt.legend(loc="lower right")
    plt.show()