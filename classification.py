"""
A simple script that demonstrates how we can use grid search to set the parameters of a classifier
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import re


REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
STOPWORDS = set(stopwords.words("english"))

# read the reviews and their polarities from a given file
def loadData(fname):
    reviews = []
    labels = []
    f = open(fname)
    for line in f:
        review, rating = line.strip().split(",")
        reviews.append(review.lower())
        labels.append(rating)
    f.close()
    return reviews, labels


def loadTest(fname):
    reviews = []
    f = open(fname)
    for line in f:
        review = line.strip()
        reviews.append(review.lower())
    f.close()
    return reviews


def clean_text(text):

    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(" ", text)
    text = BAD_SYMBOLS_RE.sub("", text)
    text = text.replace("x", "")
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text


def classifier(testFile):
    rev_train, labels_train = loadData("jobs.csv")
    rev_test = loadTest(testFile)

    # for i in range(0, len(rev_train)):
    #     rev_train[i] = clean_text(rev_train[i])
    rev_train = list(map(lambda x: clean_text(x), rev_train))

    # for i in range(0, len(rev_test)):
    #     rev_test[i] = clean_text(rev_test[i])
    rev_test = list(map(lambda x: clean_text(x), rev_test))

    # Build a counter based on the training dataset
    counter = CountVectorizer(stop_words=stopwords.words("english"))
    counter.fit(rev_train)

    # count the number of times each term appears in a document and transform each doc into a count vector
    counts_train = counter.transform(rev_train)  # transform the training data
    counts_test = counter.transform(rev_test)  # transform the testing data

    model1 = MultinomialNB(alpha=0.8, fit_prior=True, class_prior=None)
    model2 = LogisticRegression(solver="lbfgs")
    model3 = MLPClassifier(
        hidden_layer_sizes=(14,),
        activation="relu",
        solver="adam",
        batch_size="auto",
        random_state=11,
        max_iter=8,
        warm_start=True,
        learning_rate_init=0.001,
        learning_rate="constant",
        alpha=0.01,
    )

    model4 = model = sklearn.neural_network.MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=1000,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
    )

    predictors = [("Classify", model2)]

    clf = VotingClassifier(predictors)

    # train all classifier on the same datasets
    clf.fit(counts_train, labels_train)
    # (xtrain,ytrain)

    # use hard voting to predict (majority voting)
    pred = clf.predict(counts_test)

    # xtest

    # print accuracy
    # print(accuracy_score(pred, labels_test))

    fw = open("predicted_jobs.csv", "w", encoding="utf8")
    writer = csv.writer(fw, lineterminator="\n")
    j = 0
    i = 0
    while j < len(rev_test) and i < len(pred):
        writer.writerow([rev_test[j], pred[i]])
        j = j + 1
        i = i + 1

    fw.close()


"""
print('\n-- Training data --')
predictions = clf.predict(counts_train)
accuracy = sklearn.metrics.accuracy_score(labels_train, predictions)
print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
print('Classification Report:')
print(sklearn.metrics.classification_report(labels_train, predictions))
print('Confusion Matrix:')
print(sklearn.metrics.confusion_matrix(labels_train, predictions))
print('')
"""

# print("\n---- Test data ----")
# predictions = clf.predict(counts_test)
# accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
# print("Accuracy: {0:.2f}".format(accuracy * 100.0))
# print("Classification Report:")
# print(sklearn.metrics.classification_report(labels_test, predictions))
# print("Confusion Matrix:")
# print(sklearn.metrics.confusion_matrix(labels_test, predictions))


"""
for i in range(100):
	print("X=%s, Predicted=%s /n/n" % (rev_test[i].split('-')[0], predictions[i]))
   """
"""
for i in range(3):
	print(counts_test[i], predictions[i])"""

"""EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['Description'].values)
word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))


X = tokenizer.texts_to_sequences(data['Description'].values)
#max sequence length is 250(change)
X = pad_sequences(X, maxlen=250)
#print('Shape of data tensor:', X.shape)


Y = pd.get_dummies(data['Query']).values
#print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 42)

train = pd.concat([X_train, Y_train])
test = pd.concat([X_test, Y_test])

train.to_csv(index=False)
test.to_csv(index=False)

"""
"""
model = Sequential()
model.add(Embedding(50000, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


history = model.fit(X_train, Y_train, epochs=5, batch_size=100,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
accr = model.evaluate(X_test,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

"""
"""
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();"""

"""
KNN_classifier = KNeighborsClassifier()
LREG_classifier = LogisticRegression(solver="liblinear")
DT_classifier = DecisionTreeClassifier()
RF_classifier = RandomForestClassifier()
MLP_classifier = MLPClassifier(
    hidden_layer_sizes=(14,),
    batch_size="auto",
    random_state=11,
    max_iter=8,
    learning_rate_init=0.001,
    alpha=0.01,
)


predictors = [
    ("knn", KNN_classifier),
    ("lreg", LREG_classifier),
    ("dt", DT_classifier),
    ("rf", RF_classifier),
    ("MLP", MLP_classifier),
]

VT = VotingClassifier(predictors)


# =======================================================================================
# build the parameter grid
KNN_grid = [
    {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17],
        "weights": ["uniform", "distance"],
    }
]

# build a grid search to find the best parameters
gridsearchKNN = GridSearchCV(KNN_classifier, KNN_grid, cv=5)

# run the grid search
gridsearchKNN.fit(counts_train, labels_train)

# =======================================================================================

# build the parameter grid
DT_grid = [
    {"max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "criterion": ["gini", "entropy"]}
]

# build a grid search to find the best parameters
gridsearchDT = GridSearchCV(DT_classifier, DT_grid, cv=5)

# run the grid search
gridsearchDT.fit(counts_train, labels_train)

# =======================================================================================

# build the parameter grid
LREG_grid = [{"C": [0.5, 1, 1.5, 2], "penalty": ["l1", "l2"]}]

# build a grid search to find the best parameters
gridsearchLREG = GridSearchCV(LREG_classifier, LREG_grid, cv=5)

# run the grid search
gridsearchLREG.fit(counts_train, labels_train)

# =======================================================================================
# build the parameter grid
RF_grid = [
    {
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "max_features": ["auto", "sqrt", "log2"],
    }
]

# build a grid search to find the best parameters
gridsearchRF = GridSearchCV(RF_classifier, RF_grid, cv=5)

# run the grid search
gridsearchRF.fit(counts_train, labels_train)

# =======================================================================================
# build the parameter grid
MLP_grid = [
    {
        "warm_start": [True, False],
        "learning_rate": ["constant", "invscaling", "adaptive"],
    }
]

# build a grid search to find the best parameters
gridsearchMLP = GridSearchCV(MLP_classifier, MLP_grid, cv=5)

# run the grid search
gridsearchMLP.fit(counts_train, labels_train)


VT.fit(counts_train, labels_train)

# use the VT classifier to predict
predicted = VT.predict(counts_test)

# print the accuracy
print(accuracy_score(predicted, labels_test))
"""

"""
USE THIS IF YOU WANT TO SEE THE ACCURACY FOR EACH PARAM CONFIGURATION IN A GRID
#print the score for each parameter setting
for params, mean_score, scores in gridsearchKNN.grid_scores_:
    print params, mean_score
"""

classifier("test.csv")