from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
import pandas as pd


def extract_tf_idf(base_words):
    tf_idf = base_words.apply(lambda x: tf_idf(x))
    return tf_idf.tolist()


def tf_idf(sentence):
    sentence = literal_eval(sentence)
    sentence_str = [" ".join(sentence)]
    vectorizer = TfidfVectorizer()

    tf_idf = vectorizer.fit_transform(sentence_str).data
    words = vectorizer.get_feature_names()

    tf_idf_not_sorted = []
    for word in sentence:
        if word not in words:
            tf_idf_not_sorted.append(0)
        else:
            word_idx = words.index(word)
            tf_idf_not_sorted.append(tf_idf[word_idx])
    return tf_idf_not_sorted


def calculate_words_length(base_words):
    normalized_words_len = base_words.apply(lambda x: get_word_length(x))
    return normalized_words_len.tolist()


def get_word_length(words_list):
    words_len = [len(word) for word in literal_eval(words_list)]
    max_len = max(words_len)
    normalized_words_len = [word_len / max_len for word_len in words_len]
    return normalized_words_len


def get_words_numbers(ctags):
    numbers = ctags.apply(lambda x: get_word_number(x))
    return numbers.tolist()


def get_word_number(ctags):
    numbers = []
    for ctag in literal_eval(ctags):
        c = ctag.split(':')
        if 'sg' in c:
            numbers.append('singular')
        elif 'pl' in c:
            numbers.append('plural')
        else:
            numbers.append('empty')
    return numbers


def get_words_degrees(ctags):
    degrees = ctags.apply(lambda x: get_word_degree(x))
    return degrees.tolist()


def get_word_degree(ctags):
    degrees = []
    for ctag in literal_eval(ctags):
        c = ctag.split(':')
        if 'pos' in c:
            degrees.append('positive') # stopień równy
        elif 'comp' in c:
            degrees.append('comparative') # stopień wyższy
        elif 'sup' in c:
            degrees.append('superlative') # stopień najwyższy
        else:
            degrees.append('empty')
    return degrees


def get_parts_of_speech(ctags):
    parts_of_speech = ctags.apply(lambda x: get_part_of_speech(x))
    return parts_of_speech.tolist()


def get_part_of_speech_next_word(ctag):
    parts_of_speech_next = []
    parts_of_speech_words = get_part_of_speech(ctag)
    for idx in range(0, len(parts_of_speech_words)-1):
        parts_of_speech_next.append(parts_of_speech_words[idx+1])
    parts_of_speech_next.append("empty")
    return parts_of_speech_next


def get_parts_of_speech_next(ctags):
    next_pos = ctags.apply(lambda x: get_part_of_speech_next_word(x))
    return next_pos.tolist()


def get_part_of_speech_previous_word(ctag):
    parts_of_speech_previous = ["empty"]
    parts_of_speech_words = get_part_of_speech(ctag)
    for idx in range(1, len(parts_of_speech_words)):
        parts_of_speech_previous.append(parts_of_speech_words[idx - 1])
    return parts_of_speech_previous


def get_parts_of_speech_previous(ctags):
    previous_pos = ctags.apply(lambda x: get_part_of_speech_previous_word(x))
    return previous_pos.tolist()


def get_part_of_speech(ctags):
    parts = []
    verbs = ['fin', 'beddzie', 'aglt', 'praet', 'impt', 'imps', 'inf', 'pcon', 'pant', 'ger', 'pact', 'ppas', 'winien']
    adjectives = ['adj', 'adja', 'adjp'] # przymiotnik
    pronouns = ['ppron12', 'ppron3', 'siebie'] # zaimek

    for ctag in literal_eval(ctags):
        c = ctag.split(':')
        if 'subst' in c:
            parts.append('noun')
        elif 'adv' in c:
            parts.append('adverb') # przysłówek
        elif 'prep' in c:
            parts.append('prep') # przyimek
        elif len(set(c).intersection(verbs)) > 0:
            parts.append('verb')
        elif len(set(c).intersection(adjectives)) > 0:
            parts.append('adjective')
        elif len(set(c).intersection(pronouns)) > 0:
            parts.append('pronoun')
        else:
            parts.append('empty')
    return parts


def get_first_words_occurence(base_words):
    normalized_words_occ_idx = base_words.apply(lambda x: get_occurences(x))
    return normalized_words_occ_idx.tolist()


def get_occurences(words_list):
    words_idx = [words_list.index(word) for word in words_list]
    normalized_words_idx = [word_idx / len(words_idx) for word_idx in words_idx]
    return normalized_words_idx


def get_second_previous_word(base_words):
    result = []
    for words_list in base_words:
        previous_words = ['']
        previous_words.append('')
        words_list = literal_eval(words_list)
        for idx in range(2, len(words_list)):
            previous_words.append(words_list[idx - 2])
        result.append(previous_words)
    return result


def get_previous_words(base_words):
    result = []
    for words_list in base_words:
        previous_words = ['']
        words_list = literal_eval(words_list)
        for idx in range(1, len(words_list)):
            previous_words.append(words_list[idx-1])
        result.append(previous_words)
    return result


def get_second_next_words(base_words):
    result = []
    for words_list in base_words:
        next_words = []
        words_list = literal_eval(words_list)
        for idx in range(0, len(words_list)-2):
            next_words.append(words_list[idx + 2])
        next_words.append('')
        next_words.append('')
        result.append(next_words)
    return result


def get_next_words(base_words):
    result = []
    for words_list in base_words:
        next_words = []
        words_list = literal_eval(words_list)
        for idx in range(0, len(words_list)-1):
            next_words.append(words_list[idx + 1])
        next_words.append('')
        result.append(next_words)
    return result


def get_previous_next_words(base_words):
    previous_next = []
    previous_words = get_previous_words(base_words)
    next_words = get_next_words(base_words)

    for (previous_line, next_line) in zip(previous_words, next_words):
        current_next_tmp = []
        for pw, nw in zip(previous_line, next_line):
            current_next_tmp.append([pw, nw])
        previous_next.append(current_next_tmp)
    return previous_next


def get_current_next_words(base_words):
    current_next = []
    next_words = get_next_words(base_words)

    for (bw_line, next_line) in zip(base_words, next_words):
        current_next_tmp = []
        bw_list = literal_eval(bw_line)
        for cw, nw in zip(bw_list, next_line):
            current_next_tmp.append([cw, nw])
        current_next.append(current_next_tmp)
    return current_next


def create_features_list(dataset):
    features = []

    tf_idf = extract_tf_idf(dataset['base_words_list'])
    length = calculate_words_length(dataset['base_words_list'])
    occurences = get_first_words_occurence(dataset['base_words_list'])
    previous = get_previous_words(dataset['base_words_list'])
    second_previous = get_second_previous_word(dataset['base_words_list'])
    next = get_next_words(dataset['base_words_list'])
    second_next = get_second_next_words(dataset['base_words_list'])
    numbers = get_words_numbers(dataset['ctag'])
    degrees = get_words_degrees(dataset['ctag'])
    parts_of_speech = get_parts_of_speech(dataset['ctag'])
    parts_of_speech_next = get_parts_of_speech_next(dataset['ctag'])
    parts_of_speech_previous = get_parts_of_speech_previous(dataset['ctag'])
    current_next_words = get_current_next_words(dataset['base_words_list'])
    previous_next_words = get_previous_next_words(dataset['base_words_list'])

    for idx1 in range(0, len(tf_idf)):
        feature = []
        for idx2 in range(0, len(tf_idf[idx1])):
            feature.append({'LENGTH': length[idx1][idx2], 'TF-IDF': tf_idf[idx1][idx2],
                            'FIRST-OCCUR': occurences[idx1][idx2], 'PREVIOUS': previous[idx1][idx2], 'SECOND_PREVIOUS': second_previous[idx1][idx2],
                            'NEXT': next[idx1][idx2], 'SECOND_NEXT': second_next[idx1][idx2], 'NUMBER': numbers[idx1][idx2], 'DEGREE': degrees[idx1][idx2],
                            'POS': parts_of_speech[idx1][idx2], 'POS_NEXT': parts_of_speech_next[idx1][idx2],
                            'POS_PREVIOUS': parts_of_speech_previous[idx1][idx2], 'CURRENT_NEXT': current_next_words[idx1][idx2],
                            'PREVIOUS_NEXT': previous_next_words[idx1][idx2]})
        features.append(feature)

    dataset['features'] = features
    return dataset
