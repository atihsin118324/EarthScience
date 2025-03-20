一、功能：1.熱門項目2.熱門字詞+詞雲3.搜尋引擎推薦詞4.推薦同類型熱門用戶（可選擇同地區優先推薦）

二、必要的前置操作： 連線資料庫：tool.py 更改connect_db() 第12和13行的本機資料庫密碼和資料庫名稱
三、文件說明：
 basic.py 為前三個功能的程式碼
 suggest.py為第四功能主程式碼
 tool.py主要程式碼用於的函數
 create_model.py由維基百科的文檔訓練詞向量，輸出的模型放在keep_wiki_all/word2vec.model
 create_SearchDict.py由上述模型對熱門用戶(有百萬粉絲)的微博發文做特徵標記，輸出標記字典dictionary/UserTagDict.json，再進一步做相關用戶推薦，輸出推薦用戶字典dictionary/SimilarityDict.json，兩個字典用於實現第四功能
 其他做分詞相關的小檔案：user_dict.txt、cn_stopwords.txt
 詞雲圖片WordCloud.jpg (由basic.py生成)
四、參數設定：
 basic.py: line14抓取年份（預設2011）、line16抓取幾個資料（預設1000）、line22顯示幾大熱門地區（預設10個）、line33顯示幾大熱門詞（預設6個）、line52搜尋引擎推薦詞的字數上限（預設值7個）、line61個搜尋引擎推薦詞的字數上限（預設值）
 line64機率門值(數值高於搜尋引擎推薦詞數好奇越短，預設0.8)、line70欲查詢的輸入內容(預設台北)
 suggest.py: line11和line37顯示推薦用戶的數量（預設50和11）、line11是否優先推薦同地區用戶search_near（預設False）、line33虛擬主角的id（必須在SimilarityDict中的id，前面SimilarityDict指輸出熱門用戶（有數百萬粉絲，Dict-Similarity[Similarity”
五、字典文檔格式：
 UserTagDict: {user_id : user('name', 'loc', 'id', 'gender', 'followers_count', 'description', 'tags', 'texts' )} （並已經以followers_count做排序，第一個為最熱門使用者）
 SimilarityDict: {myid : [others('name', 'loc', 'id', 'gender', 'followers_count', 'description', 'tags', 'texts'), similarity_dist]} （並且已經以similarity_dist做排序，第一個為與本人最相關的使用者）
六、簡易使用說明：
 設定basic.py：line70欲查詢的輸入內容（預設台北）
 suggest.py: line33虛擬主角的id(必須在SimilarityDict中的id，前面SimilarityDict指輸出熱門用戶(有數百萬粉絲，預設為SimilarityDict[1]也就是第二人)