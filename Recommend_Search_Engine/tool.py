import mysql.connector
import re
from collections import Counter, namedtuple


def connect_db():
    try:
        print("正在連接mysql服務器")
        database = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="******",
            db="******"
        )
        print("連接成功")

    except Exception as e:
        database.close()
        print(str(e))
    return database


def scenic_spot(documents, n=10):
    locations = []
    for text in documents:
        text = re.sub(" ", "", text)
        try:
            p = re.findall("我在[^，！。；!]*?\#.*?\#", text)[0]
            p = re.split('#', p)[1]
            p = re.sub("我在这里:", '', p)
            p = re.sub("\(.*", '', p)
            p = re.sub("\（.*", '', p)
            p = re.sub("-.*", '', p)
            locations.append(p)
        except:
            pass
        try:
            p = re.findall(r"我在[^\#]*?[，！。；!]", text)[0]
            p = re.sub("我在(这里)?", '', p)
            p = re.sub("(http).*?", '', p)
            p = re.sub(r"^\（", '', p)
            p = re.sub(r"^\(", '', p)
            p = re.sub("\(.*", '', p)
            p = re.sub("\（.*", '', p)
            p = re.sub("-.*", '', p)
            p = re.sub(r"[\W]", '', p)
            locations.append(p)
        except:
            pass
    locations_counter = Counter(locations)
    scenic_spot = locations_counter.most_common(n)
    return scenic_spot


def only_chinese(documents):
    f_clean = [re.sub(r"[^\u4e00-\u9fa5]", '', seg) for seg in documents]
    return f_clean


def remove_stopwords(sw_file, words):
    f = open(sw_file, 'r', encoding="utf-8")
    stopwords = list(set(f.rstrip() for f in f.readlines()))
    f.close()
    without_stopwords = [w for w in words if w not in stopwords]
    return without_stopwords


def ngram(documents, n):
    ngram_numerator = []
    ngram_denominator = []
    ngram_prediction = {}
    Word = namedtuple('Word', ['word', 'prob'])

    for doc in documents:
        split_words = list(doc)
        [ngram_numerator.append(tuple(split_words[i:i + n])) for i in
         range(len(split_words) - n)]
        [ngram_denominator.append(tuple(split_words[i:i + n - 1])) for i in
         range(len(split_words) - (n - 1))]
    ngram_numerator_count = Counter(ngram_numerator)
    ngram_denominator_count = Counter(ngram_denominator)

    for key in ngram_numerator_count:
        word = ''.join(key[:n - 1])
        if word not in ngram_prediction:
            ngram_prediction.update({word: set()})
        next_word_prob = ngram_numerator_count[key] / ngram_denominator_count[key[:n - 1]]
        w = Word(key[-1], '{:.3g}'.format(next_word_prob))
        ngram_prediction[word].add(w)

    for word, nextwords in ngram_prediction.items():
        ngram_prediction[word] = sorted(nextwords, key=lambda x: x.prob,
                                        reverse=True)
    return ngram_prediction


