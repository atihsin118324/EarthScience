import logging
from gensim.corpora.wikicorpus import extract_pages,filter_wiki
import bz2file
wiki = extract_pages(bz2file.open('zhwiki-20220601-pages-articles.xml.bz2'))
from tqdm import tqdm
from opencc import OpenCC
import jieba
import re
from gensim.models import word2vec, Word2Vec

def wiki_replace(d):
    cc = OpenCC('s2t')
    s = d[1]
    s = re.sub(':*{\|[\s\S]*?\|}', '', s)
    s = re.sub('<gallery>[\s\S]*?</gallery>', '', s)
    s = re.sub('(.){{([^{}\n]*?\|[^{}\n]*?)}}', '\\1[[\\2]]', s)
    s = filter_wiki(s)
    s = re.sub('\* *\n|\'{2,}', '', s)
    s = re.sub('\n+', '\n', s)
    s = re.sub('\n[:;]|\n +', '\n', s)
    s = re.sub('\n==', '\n\n==', s)
    s = u'【' + d[0] + u'】\n' + s
    return cc.convert(s).strip()

i=0
f = open('wiki.txt', 'w', encoding='utf-8')
w = tqdm(wiki, desc=u'已獲取0篇文章')
for d in w:
    if not re.findall('^[a-zA-Z]+:', d[0]) and d[0] and not re.findall(u'^#', d[1]):
        s = wiki_replace(d)
        f.write(s+'\n\n\n')
        i += 1
        if i % 100 == 0:
            w.set_description(u'已獲取%s篇文章'%i)

f.close()

#
stopword_set = set()
with open('cn_stopwords.txt', 'r', encoding='utf-8') as stopwords:
    for stopword in stopwords:
        stopword_set.add(stopword.strip('\n'))

output = open('wiki_seg.txt', 'w', encoding='utf-8')
with open('wiki.txt', 'r', encoding='utf-8') as content:
    for texts_num, line in enumerate(content):
        line = line.strip('\n')
        words = jieba.cut(line, cut_all=False)
        for word in words:
            if word not in stopword_set:
                output.write(word + ' ')
        output.write('\n')

        if (texts_num + 1) % 10000 == 0:
            logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
output.close()

sentences = word2vec.LineSentence("wiki_seg.txt")
model = Word2Vec(sentences, vector_size=250)
model.save("word2vec.model")