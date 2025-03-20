from tool import *
import jieba.analyse
import jieba
from gensim.models import word2vec
import json

# 從資料庫提取大用戶
jieba.load_userdict('user_dict.txt')
StopWordsFile = 'cn_stopwords.txt'

db = connect_db()
cursor = db.cursor()
sql = "SELECT id, screen_name, location, description, followers_count, gender FROM travel_poi_userinfo_suzhou  " \
      "WHERE followers_count > 1000000 ;"
cursor.execute(sql)
result_sql = cursor.fetchall()

id = [str(seg[0]) for seg in result_sql]
screen_name = [seg[1] for seg in result_sql]
location = [seg[2] for seg in result_sql]
description = [seg[3]) for seg in result_sql]
followers_count = [seg[4] for seg in result_sql]
gender = [seg[5] for seg in result_sql]

# 1.提取人物標籤
UserTagDict = {}
User = namedtuple('User',
                  ['name', 'loc', 'id', 'gender', 'followers_count', 'description',
                   'tags', 'texts'])

for n in range(len(id)):
    sql2 = f"SELECT text FROM travel_poi_weibos_suzhou WHERE userid = {id[n]};"
    cursor.execute(sql2)
    result_sql2 = cursor.fetchall()
    if len(result_sql2):
        result_sql2 = [str(seg[0]) for seg in result_sql2]
        ChineseContent = only_chinese(result_sql2)
        ChineseContentAll = ''.join(ChineseContent)
        Words = jieba.cut(ChineseContentAll, cut_all=False)
        CleanWords = remove_stopwords(StopWordsFile, Words)
        CleanContent = ' '.join(CleanWords)

        Tags = jieba.analyse.extract_tags(CleanContent, 6)

        p = User(screen_name[n], location[n], id[n], gender[n], followers_count[n],
                 description[n], Tags, CleanContent)
        UserTagDict.update({id[n]: p})

db.close()

UserTagDict = sorted(UserTagDict.items(), key=lambda d: d[1].followers_count, reverse=False)
UserTagDict = dict(UserTagDict)

UserTagDict_json = json.dumps(UserTagDict, sort_keys=False, ensure_ascii=False, indent=3)
with open('dictionary/UserTagDict.json', 'w', encoding="utf-8") as f:
    f.write(UserTagDict_json)
print('UserTagDict.json字典儲存成功: dictionary/UserTagDict.json')

# 2.找相似用戶
SimilarityDict = {}
print('開始下載model')
model = word2vec.Word2Vec.load("keep_wiki_all/word2vec.model")
print('下載model成功')

for target_id, target in UserTagDict.items():
    similar_compare = []
    for c_id, compare in UserTagDict.items():
        if c_id == target_id or len(target.tags) < 6 or len(compare.tags) < 6:
            continue
        dist = model.wv.wmdistance(target.texts.split(), compare.texts.split())
        if dist < 1.5:
            similar_compare.append([compare, round(dist, 3)])
    SimilarityDict.update({target_id: similar_compare})

for p, similar in SimilarityDict.items():
    SimilarityDict[p] = sorted(similar, key=lambda d: d[1], reverse=False)
SimilarityDict = dict(SimilarityDict)

Similarity_json = json.dumps(SimilarityDict, sort_keys=False, ensure_ascii=False, indent=3)
with open('dictionary/SimilarityDict.json', 'w', encoding="utf-8") as f:
    f.write(Similarity_json)
print('SimilarityDict.json字典儲存成功: dictionary/SimilarityDict.json')