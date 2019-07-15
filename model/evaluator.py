from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from ast import literal_eval


labels = {'I': 0, 'O': 1, 'B': 2, 'S': 3}


def label_data(data):
    result = []
    for row in data:
        try:
            for x in literal_eval(row):
                result.append(labels[x])
        except:
            for x in row:
                result.append(labels[x])
    return result


def get_confusion_matrix(y_true, y_pred):
    y_true = label_data(y_true)
    y_pred = label_data(y_pred)
    return confusion_matrix(y_true, y_pred)


def evaluate_each_class(y_true, y_pred):
    print('Evaluating...')
    y_true = y_true.tolist()
    y_true = label_data(y_true)
    y_pred = label_data(y_pred)
    return (accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average=None),
            recall_score(y_true, y_pred, average=None), f1_score(y_true, y_pred, average=None))


def evaluate_macro(y_true, y_pred):
    print('Evaluating...')
    y_true = y_true.tolist()
    y_true = label_data(y_true)
    y_pred = label_data(y_pred)
    return (accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='macro'))


def hard_evaluation(keywords_true, keywords_pred):
    TP = 0.0
    FP = 0.0
    FN = 0.0

    for keywords_true_row, keywords_pred_row in zip(keywords_true, keywords_pred):
        keywords_true_row = literal_eval(keywords_true_row)
        for k_pred in keywords_pred_row:
            if k_pred in keywords_true_row:
                TP += 1
                keywords_true_row.remove(k_pred)
            else:
                FP += 1
        FN += len(keywords_true_row)

    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if (2 * TP + FP + FN) == 0:
        f_score = 0
    else:
        f_score = 2 * TP / (2 * TP + FP + FN)
    return precision, recall, f_score


def soft_evaluation(keywords_true, keywords_pred):
    TP = 0.0
    FP = 0.0
    FN = 0.0

    for keywords_true_row, keywords_pred_row in zip(keywords_true, keywords_pred):
        keywords_true_row = literal_eval(keywords_true_row)
        keywords_true_row_splitted = [x for i in keywords_true_row for x in i.split()]
        keywords_pred_row_splitted = [x for i in keywords_pred_row for x in i.split()]

        for k_pred in keywords_pred_row_splitted:
            if k_pred in keywords_true_row_splitted:
                TP += 1
                keywords_true_row_splitted.remove(k_pred)
            elif k_pred in keywords_true_row_splitted:
                TP += 1
                FP += 1
            else:
                FP += 1
        FN += len(keywords_true_row_splitted)

    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if (2 * TP + FP + FN) == 0:
        f_score = 0
    else:
        f_score = 2 * TP / (2 * TP + FP + FN)
    return precision, recall, f_score
