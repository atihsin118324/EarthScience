import jieba.analyse
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tool import *

jieba.load_userdict('user_dict.txt')
StopWordsFile = 'cn_stopwords.txt'

db = connect_db()
cursor = db.cursor()
sql = "SELECT text FROM geotaggedweibo WHERE time LIKE '2011%' ;"
cursor.execute(sql)
result_sql = cursor.fetchmany(1000)
result_sql = [str(seg[0]) for seg in result_sql]
db.close()

# 1.打卡熱門地區
ScenicSpot = scenic_spot(result_sql, 10)
print('ScenicSpot:', ScenicSpot)

ChineseContent = only_chinese(result_sql)
ChineseContentAll = ''.join(ChineseContent)
Words = jieba.cut_for_search(ChineseContentAll)
CleanWords = remove_stopwords(StopWordsFile, Words)
CleanContent = ' '.join(CleanWords)

# 2.1熱門字
Tags = jieba.analyse.extract_tags(CleanContent, 6)
print('Buzzwords:', ",".join(Tags))

# 2.2詞雲
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      min_font_size=10,
                      font_path='simhei.ttf').generate(CleanContent)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.savefig(f'WordCloud.jpg')

# 3.搜尋引擎推薦字
LargestNgram = 7
Predict_dict = dict(zip(range(2, LargestNgram), list(ngram(ChineseContent, N) for N in range(2, LargestNgram))))

def maker(text):
    n = len(text) + 1
    ngram_pred = Predict_dict[n]
    try:
        next_words = list(ngram_pred[text])[:6]
        for next_word in next_words:
            new_text = text + next_word.word if float(
                next_word.prob) > 0.8 else text
    except:
        new_text = text
    return new_text

Input = '台北'
print('Search:', Input)
Output = []

if Input in Predict_dict[len(Input)+1].keys():
    NextWords = list(Predict_dict[len(Input)+1][Input])[:5]
    Text = [Input + nw.word for nw in NextWords]
    for i in range(LargestNgram - len(Input) - 2):
        Text = map(maker, Text)
    print('Show:')
    for i, word in enumerate(Text):
        print(word)
else:
    print('Show:', Input)