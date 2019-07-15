import xml.etree.ElementTree as ET
import glob
import pandas as pd


def load_processed_data(filenames=glob.glob(".\\kpwr-1.1\\*\\result.csv")):
    df_list = [pd.read_csv(file, engine='python', encoding='utf-8') for file in filenames]
    data = pd.concat(df_list)
    return data


def load_test_data(filename=".\\data\\result-IOBS.csv"):
    return pd.read_csv(filename, encoding='utf-8')


def load_text_data(file):
    keywords = ""
    text = ""
    with open(file, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                keywords = line.rstrip()
            else:
                text += line.rstrip()
    return keywords, text


def read_origin(xml):
    with open(xml, 'r', encoding="utf-8") as content:
        tree = ET.parse(content)
        root = tree.getroot()
        str_list_origin = []
        str_features = []
        for word in root.iter('tok'):
            str_list_origin.append(word.find('orth').text)
            str_features.append(word.find('lex').find('ctag').text)
        origin_text = ' '.join(str_list_origin)
    return origin_text, str_features


def read_file(xml):
    with open(xml, 'r', encoding="utf-8") as content:
        tree = ET.parse(content)
        root = tree.getroot()

        str_list_origin = []
        str_list_base = []
        str_features = []
        for word in root.iter('tok'):
            str_list_origin.append(word.find('orth').text)
            str_list_base.append(word.find('lex').find('base').text)
            str_features.append(word.find('lex').find('ctag').text)
        origin_text = ' '.join(str_list_origin)
        base_text = ' '.join(str_list_base)
    return origin_text, base_text, str_features
