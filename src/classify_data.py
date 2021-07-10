import os
import joblib
from sklearn.ensemble import RandomForestClassifier
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

import config


def prepare_class_data(y, fit_from_scratch=False):
    """

    This will create a one-hot encoded representation of each list of tags. The length of each vector 
    is equal to the number of tags in the dataset. 

    TODO: Consider removing tags (and potentially texts, if they only have such tag(s)) that appear 
    less than k times.

    """
    if not os.path.exists(config.BINARIZER_PKL) or fit_from_scratch:
        label_binarizer = MultiLabelBinarizer()
        class_vectors = label_binarizer.fit_transform(y)
        joblib.dump(label_binarizer, config.BINARIZER_PKL)
    else:
        label_binarizer = joblib.load(config.BINARIZER_PKL)
        class_vectors = label_binarizer.transform(y)
    return class_vectors


def extract_features(texts, fit_from_scratch=False):
    text_vectors = []
    if not os.path.exists(config.VECTORIZER_PKL) or fit_from_scratch:
        vectorizer = TfidfVectorizer(texts, ngram_range=(1, 3), stop_words='english', max_features=10000)
        text_vectors = vectorizer.fit_transform(texts)
        joblib.dump(vectorizer, config.VECTORIZER_PKL)
    else:
        vectorizer = joblib.load(config.VECTORIZER_PKL)
        text_vectors = vectorizer.transform(texts)

    return text_vectors


def train_model_one_vs_rest(train_vectors, train_labels):
    model = RandomForestClassifier()
    clf = MultiOutputClassifier(model)
    clf.fit(train_vectors, train_labels)
    return clf


def classify_texts(test_vectors, clf):
    predictions = clf.predict(test_vectors)
    return predictions


if __name__ == '__main__':
    print("Loading data...")
    data = joblib.load(config.FILENAME_PKL)
    # data = data[:10000]
    X = [a['body'] for a in data]
    y = [a['keywords'] for a in data]

    print("Vectorizing class labels...")
    y_vectors = prepare_class_data(y)

    print("Performing train test split...")
    trainX, testX, trainY, testY = train_test_split(X, y_vectors)

    print("Extracting features from text...")
    train_vectors_X = extract_features(trainX, fit_from_scratch=True)
    test_vectors_X = extract_features(testX)

    print("Training model...")
    clf = train_model_one_vs_rest(train_vectors_X, trainY)

    print("Evaluating model...")
    predictions = classify_texts(test_vectors_X, clf)
    print(classification_report(testY, predictions))

    # TODO: Instead of classification report, write results of predictions on test data to file
    # Write the predicted vs actual tags to the file, so we can see what's going on
