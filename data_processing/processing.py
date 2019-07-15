import xml.etree.ElementTree as ET
import configparser
import glob
import pandas as pd
import string
from data_preprocessing.tagger import tagging, lemmatization
from data_loader.data_loader import read_origin, read_file, load_text_data
from stop_words import get_stop_words


def process_data_IOBS(xml, keywords):
    origin_text, base_text, ctag = read_file(xml)
    base_keywords = find_base_form(lemmatization(keywords))

    # words to lowercase
    origin_words_list = list_to_lowercase(origin_text.split())
    base_words_list = list_to_lowercase(base_text.split())
    base_keywords = list_to_lowercase(base_keywords.split(", "))
    origin_keywords = list_to_lowercase(keywords.split(", "))

    # remove punctuations
    origin_words_list = remove_punctuation(origin_words_list)
    base_words_list = remove_punctuation(base_words_list)
    origin_keywords = remove_punctuation(origin_keywords)

    origin_keywords_in_text = get_keywords_from_text(origin_keywords, origin_words_list)
    base_keywords_in_text = get_keywords_from_text(base_keywords, base_words_list)
    labels_base, labels_origin = labelling_texts(origin_words_list, base_words_list, keywords, base_keywords_in_text)

    return origin_text, base_text, keywords, base_keywords, base_keywords_in_text, origin_keywords_in_text, \
           base_words_list, origin_words_list, ctag, labels_base, labels_origin


def process_data_IOB(xml, keywords):
    origin_text, base_text, ctag = read_file(xml)
    base_keywords = find_base_form(lemmatization(keywords))

    # words to lowercase
    origin_words_list = list_to_lowercase(origin_text.split())
    base_words_list = list_to_lowercase(base_text.split())
    base_keywords = list_to_lowercase(base_keywords.split(", "))
    origin_keywords = list_to_lowercase(keywords.split(", "))

    # remove punctuations
    origin_words_list = remove_punctuation(origin_words_list)
    base_words_list = remove_punctuation(base_words_list)
    origin_keywords = remove_punctuation(origin_keywords)

    origin_keywords_in_text = get_keywords_from_text(origin_keywords, origin_words_list)
    base_keywords_in_text = get_keywords_from_text(base_keywords, base_words_list)
    labels_base, labels_origin = labelling_texts(origin_words_list, base_words_list, keywords, base_keywords_in_text)

    return origin_text, base_text, keywords, base_keywords, base_keywords_in_text, origin_keywords_in_text, \
           base_words_list, origin_words_list, ctag, labels_base, labels_origin


def create_data_from_text(filenames=glob.glob(".\\data\\*.xml")):
    result = pd.DataFrame(
        columns=['original_text', 'base_text', 'keywords', 'base_keywords', 'base_keywords_in_text',
                 'origin_keywords_in_text', 'base_words_list', 'origin_words_list', 'ctag', 'label_base', 'label_origin'
                 ])

    keywords_list = ["szybkie pożyczki gotówkowe, chwilówka, pożyczka, pożyczka przez internet", "kurs pierwszej pomocy",
                     "mechanik samochodowy, warsztat samochodowy, przegląd auta",
                     "ubezpieczenia zdrowotne, ubezpieczenia prywatne", "ubezpieczenie na życie"]

    for idx, file in enumerate(filenames):
        # keywords, origin_text = load_text_data(file)
        # xml = tagging(origin_text, file.replace(".txt", "_out.xml"))
        keywords = keywords_list[idx]
        result.loc[len(result)] = process_data_IOBS(file, keywords)
    result.to_csv(path_or_buf='.\\data\\result-IOBS.csv', index=False, encoding='utf-8')


def create_data_kpwr(folder_name):
    xml_filenames = glob.glob(".\\kpwr-1.1\\" + folder_name + "\\*[0-9]_out.xml"),
    ini_filenames = glob.glob(".\\kpwr-1.1\\" + folder_name + "\\*[0-9].ini")

    result = pd.DataFrame(
        columns=['original_text', 'base_text', 'keywords', 'base_keywords', 'base_keywords_in_text',
                 'origin_keywords_in_text', 'base_words_list', 'origin_words_list', 'ctag', 'label_base', 'label_origin'
                 ])

    for xml, ini in zip(xml_filenames, ini_filenames):
        origin_text = read_origin(xml)
        xml = tagging(origin_text, xml)

        config = configparser.ConfigParser()
        config.sections()
        config.read(ini, encoding='utf-8')
        keywords = config['metadata']['keywords']
        result.loc[len(result)] = process_data_IOB(xml, keywords)

    filename = ".\\kpwr-1.1\\" + folder_name + "\\result-IOB.csv"
    result.to_csv(filename, index=False, encoding='utf-8')


def get_keywords_from_text(keywords, words_list):
    keywords_in_text = []
    for i in range(0, len(keywords)):
        if len(keywords[i].split()) > 1:
            keywords_tmp = ""
            for keyword in keywords[i].split():
                if keyword in words_list:
                    keywords_tmp += keyword + " "
            if keywords_tmp != "":
                keywords_in_text.append(keywords_tmp[:-1])
        elif keywords[i] in words_list:
            keywords_in_text.append(keywords[i])
    return keywords_in_text


def remove_punctuation(words_list):
    # remove punctation
    result = [''.join(w for w in words if w not in string.punctuation) for words in words_list]
    # remove empty strings
    result = [w for w in result if w]
    return result


def find_base_form(lemmatization):
    root = ET.fromstring(lemmatization)
    str_list_base = []
    for word in root.iter('tok'):
        str_list_base.append(word.find('lex').find('base').text)
    base_keywords = ' '.join(str_list_base)
    return base_keywords.replace(" ,", ",")


def labelling_texts(original_text, base_text, base_keywords):
    labels_base = []
    for word in base_text:
        labels_base.append(labelling_word_IOB(word, base_keywords))

    labels_origin = []
    for word in original_text:
        labels_origin.append(labelling_word_IOB(word, base_keywords))

    return labels_base, labels_origin


def labelling_word_IOBS(word, base_keywords, stop_words):
    if word in stop_words:
        return 'S'

    long_base_keywords = [word for word in base_keywords if " " in word]

    if word in base_keywords:
        return 'I'
    else:
        for item in long_base_keywords:
            splitted = item.split()
            if word in splitted:
                if splitted[0] == word:
                    return 'B'
                else:
                    return 'I'
    return 'O'


def labelling_word_IOB(word, base_keywords):
    long_base_keywords = [word for word in base_keywords if " " in word]

    if word in base_keywords:
        return 'I'
    else:
        for item in long_base_keywords:
            splitted = item.split()
            if word in splitted:
                if splitted[0] == word:
                    return 'B'
                else:
                    return 'I'
    return 'O'


def list_to_lowercase(list):
    return [x.lower() for x in list]
