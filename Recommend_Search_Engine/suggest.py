"""['name', 'loc', 'id', 'gender', 'followers_count', 'description',
                   'tags', 'texts'])"""
import json

with open('dictionary/UserTagDict.json', 'r', encoding="utf-8") as f:
    UserTagDict = json.load(f)
with open('dictionary/SimilarityDict.json', 'r', encoding="utf-8") as f:
    SimilarityDict = json.load(f)


def suggest(target_id, user_dict, similar_dict, search_near=False, n=50):
    target = user_dict[target_id]
    similar = list(similar_dict.values())[0]
    friends_suggestions = []
    candidate = []

    print('我:', target)
    if search_near:
        for friend in similar:
            if friend[0].loc == target.loc or friend[0].loc.split()[0] == target.loc:
                friends_suggestions.append(friend)

            else:
                candidate.append(friend)
        while len(friends_suggestions) < n:
            friends_suggestions.append(candidate.pop(0))
    else:
        for friend in similar[:n]:
            friends_suggestions.append(friend)
    return friends_suggestions


TargetID = list(SimilarityDict.keys())[1]

FriendsSuggestions = suggest(TargetID, UserTagDict, SimilarityDict)

for Friend in FriendsSuggestions[:11]:
    similar_dist = Friend[1]
    print(similar_dist, Friend[0][:7])

