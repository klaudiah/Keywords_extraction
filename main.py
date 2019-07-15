from data_loader.data_loader import load_processed_data
from model import CRF
from model import evaluator
from feature_extraction import create_features_list
from ast import literal_eval
from data_preprocessing.preprocessing import create_data_kpwr, create_data_from_text
from data_loader.data_loader import load_test_data
import numpy as np


def get_keywords_from_labels(words, labels):
    keywords = []
    for words_row, labels_row in zip(words, labels):
        keywords_tmp = []
        words_row = literal_eval(words_row)

        for idx in range(len(words_row)):
            if labels_row[idx] == "B" and idx+1 < len(words_row) and labels_row[idx+1] == "I":
                keywords_tmp.append(words_row[idx] + " " + words_row[idx + 1])
            elif labels_row[idx] == "I":
                if idx > 0 and labels_row[idx-1] == "B":
                    continue
                else:
                    keywords_tmp.append(words_row[idx])
        keywords.append(keywords_tmp)
    return keywords


dataset = load_processed_data()
dataset = create_features_list(dataset)

train, test = CRF.split_data(dataset)
CRF.train(train['features'], train['label_base'])
preds = CRF.test(test['features'])

keywords_true = test['base_keywords_in_text']
keywords_pred = get_keywords_from_labels(test['base_words_list'], preds)

prec_h, rec_h, f1_h = evaluator.hard_evaluation(keywords_true, keywords_pred)
prec_s, rec_s, f1_s = evaluator.soft_evaluation(keywords_true, keywords_pred)

print("Sotf evalution: Precission: %.2f, Recall: %.2f, F1Score: %.2f" % (np.mean(precision_soft_list) * 100, np.mean(recall_soft_list) * 100, np.mean(f1_soft_list) * 100))
print("Hard evalution: Precission: %.2f, Recall: %.2f, F1Score: %.2f" % (np.mean(precision_hard_list) * 100, np.mean(recall_hard_list) * 100, np.mean(f1_hard_list) * 100))


for kt, kp in zip(keywords_true, keywords_pred):
    print("Keywords true: " + str(kt))
    print("Predicted keywords: " + str(kp))
    print("\n")
