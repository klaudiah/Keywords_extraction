from pycrfsuite import Trainer, Tagger, ItemSequence
from sklearn.model_selection import train_test_split
from ast import literal_eval


def split_data(dataset):
    train, test = train_test_split(dataset)
    return train, test


def train(features, labels):
    print("Training..")
    trainer = Trainer(verbose=False)
    features = features.tolist()
    labels = labels.tolist()

    for idx in range(0, len(features)):
        trainer.append(ItemSequence(features[idx]), literal_eval(labels[idx]))
    trainer.train('crf.model')


def test(features):
    print("Testing..")
    tagger = Tagger()
    tagger.open('crf.model')
    y_pred = [tagger.tag(xseq) for xseq in features]
    return y_pred
