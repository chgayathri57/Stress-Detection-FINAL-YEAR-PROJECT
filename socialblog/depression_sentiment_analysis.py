import json
import pandas as pd
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from tweepy import OAuthHandler
import json

# from sklearn.metrics import roc_auc_score
import csv

tweets_data = []
x = []
y = []
vectorizer = CountVectorizer(stop_words='english')
y1 = []
y2 = []
y3 = []
y4 = []


def retrieveTweet(data_url):
    tweets_data_path = data_url
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue


def retrieveProcessedData(Pdata_url):
    sent = pd.read_excel(Pdata_url)
    for i in range(len(tweets_data)):
        if tweets_data[i]['id'] == sent['id'][i]:
            x.append(tweets_data[i]['text'])
            y.append(sent['sentiment'][i])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def nbTrain():
    from sklearn.naive_bayes import MultinomialNB
    start_timenb = time.time()
    train_features = vectorizer.fit_transform(x)

    actual = y

    nb = MultinomialNB()
    nb.fit(train_features, [int(r) for r in y])

    test_features = vectorizer.transform(x)
    predictions = nb.predict(test_features)
    fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

    nbscore = format(metrics.auc(fpr, tpr))
    nbscore = float(nbscore) * 100

    nb_matrix = confusion_matrix(actual, predictions)
    plt.figure()
    plot_confusion_matrix(nb_matrix, classes=[-1, 0, 1], title='Confusion Matrix For Naive Bayes Classifier')
    # plt.show()
    nbrecall = (np.mean(np.diag(nb_matrix) / np.sum(nb_matrix, axis=1))) * 100
    nbprecision = (np.mean(np.diag(nb_matrix) / np.sum(nb_matrix, axis=0))) * 100

    print("\n")

    print("Naive Bayes  Accuracy : \n", nbscore, "%")
    print("Naive Bayes  Precision : \n", nbprecision, "%")
    print("Naive Bayes  Recall : \n", nbrecall, "%")
    nbtime = round((time.time() - start_timenb), 5)
    print(" Completion Speed", nbtime)
    y1.append(nbscore)
    y1.append(nbprecision)
    y1.append(nbrecall)
    y4.append(nbtime)
    print()


def datree():
    from sklearn import tree
    start_timedt = time.time()
    train_featurestree = vectorizer.fit_transform(x)
    actual1 = y
    test_features1 = vectorizer.transform(x)
    dtree = tree.DecisionTreeClassifier()

    dtree = dtree.fit(train_featurestree, [int(r) for r in y])

    prediction1 = dtree.predict(test_features1)
    ddd, ttt, thresholds = metrics.roc_curve(actual1, prediction1, pos_label=1)
    dtreescore = format(metrics.auc(ddd, ttt))
    dtreescore = float(dtreescore) * 100

    nb_matrix = confusion_matrix(actual1, prediction1)
    plt.figure()
    plot_confusion_matrix(nb_matrix, classes=[-1, 0, 1], title='Confusion Matrix For Decision Tree Classifier')
    # plt.show()
    dtrecall = (np.mean(np.diag(nb_matrix) / np.sum(nb_matrix, axis=1))) * 100
    dtprecision = (np.mean(np.diag(nb_matrix) / np.sum(nb_matrix, axis=0))) * 100

    print("\n")

    print("Decision tree Accuracy : \n", dtreescore, "%")
    print("Decision tree Precession : \n", dtprecision, "%")
    print("Decision tree Recall : \n", dtrecall, "%")
    dttime = round((time.time() - start_timedt), 5)
    print(" Completion Speed", dttime)
    y2.append(dtreescore)
    y2.append(dtprecision)
    y2.append(dtrecall)
    y4.append(dttime)
    print()


def Tsvm():
    from sklearn.svm import SVC
    start_timesvm = time.time()
    train_featuressvm = vectorizer.fit_transform(x)
    actual2 = y
    test_features2 = vectorizer.transform(x)
    svc = SVC()

    svc = svc.fit(train_featuressvm, [int(r) for r in y])
    prediction2 = svc.predict(test_features2)
    sss, vvv, thresholds = metrics.roc_curve(actual2, prediction2, pos_label=1)
    svc = format(metrics.auc(sss, vvv))
    svc = float(svc) * 100

    nb_matrix = confusion_matrix(actual2, prediction2)
    plt.figure()
    plot_confusion_matrix(nb_matrix, classes=[-1, 0, 1], title='Confusion Matrix For SVM Classifier')
    plt.show()
    svmrecall = (np.mean(np.diag(nb_matrix) / np.sum(nb_matrix, axis=1))) * 100
    svmprecision = (np.mean(np.diag(nb_matrix) / np.sum(nb_matrix, axis=0))) * 100
    print("\n")

    print("Support vector machine Accuracy : \n", svc, "%")
    print("Support vector machine Precession : \n", svmprecision, "%")
    print("Support vector machine Recall : \n", svmrecall, "%")
    svmtime = round((time.time() - start_timesvm), 5)
    print(" Completion Speed", svmtime)
    y3.append(svc)
    y3.append(svmprecision)
    y3.append(svmrecall)
    y4.append(svmtime)

    print("")


def graph():
    x = np.arange(3)

    width = 0.25

    # plot data in grouped manner of bar type
    plt.bar(x - 0.2, y1, width, color='red')
    plt.bar(x, y2, width, color='blue')
    plt.bar(x + 0.2, y3, width, color='green')
    plt.xticks(x, ['Accuracy', 'Precision', 'Recall'])
    plt.xlabel("Labels")
    plt.ylabel("Scores")
    plt.legend(["Naive Bayes", "Decision Tree", "SVM"])
    plt.title("Performance Comparision")
    plt.show()

    # N = 3
    # ind = np.arange(N)
    # width = 0.25
    #
    # bar1 = plt.bar(ind, y1, width, color='r')
    #
    # bar2 = plt.bar(ind + width, y2, width, color='g')
    #
    # bar3 = plt.bar(ind + width * 2, y3, width, color='b')
    #
    # plt.xlabel("Labels")
    # 4 plt.ylabel('Scores')
    # plt.title("Perfomence Comparasion")
    #
    # plt.xticks(ind + width, ['Accuracy', 'Precession', 'Recall'])
    # plt.legend((bar1, bar2, bar3), ("NaiveBayes", "DecisionTree", "SVM"))
    # plt.show()


def timecmp():
    t1 = np.array([0, y4[0]])
    t2 = np.array([0, y4[1]])
    t3 = np.array([0, y4[2]])

    plt.plot(t1)
    plt.plot(t2)
    plt.plot(t3)
    # naming the y axis
    plt.ylabel('Seconds')
    plt.legend(['Naive Bayes', 'Decision Tree', 'SVM'])
    # giving a title to my graph
    plt.title('Time Evaluation')
    plt.show()


def datreeINPUT(tweet):
    loaddata()
    from sklearn import tree
    train_featurestree = vectorizer.fit_transform(x)
    dtree = tree.DecisionTreeClassifier()

    dtree = dtree.fit(train_featurestree, [int(r) for r in y])

    inputdtree = vectorizer.transform([tweet])
    predictt = dtree.predict(inputdtree)

    if predictt == 1:
        predictt = "No Stress Detection"
    elif predictt == 0:
        predictt = "Weak Stress Tendecy/Neutral"
    elif predictt == -1:
        predictt = "Strong Stress Tendency"
    else:
        print("Nothing")

    print("\n*****************")
    print(predictt)
    print("*****************")
    # keyinput(tweet)
    return predictt


def keyinput(tweet):
    with open("socialblog/ml/data/keywords.csv", 'r') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        total = 0
        for row in data:
            if len(row) > 0:
                if (tweet.__contains__(row[0])):
                    print("weightage of ", row[0], " is ", row[1])
                    total = total + int(row[1])
        print("total suicidal weightage of ", tweet, " is ", total)


def loaddata():
    retrieveTweet('socialblog/ml/data/tweetdata.txt')
    retrieveProcessedData('socialblog/ml/processed_data/output.xlsx')
