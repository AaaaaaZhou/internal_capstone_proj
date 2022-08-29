# rom pantheon import pantheon
from riot_api import ApiCaller
import asyncio

import time, re, json
from tqdm import tqdm

region = "kr"
api_key = ""
# api_key = "RGAPI-6329a2ac-a7b3-42a1-9b47-0ee799c6f2c8"

# seed_id = ["duimianxiaodai", "fragiIe", "0o0OoO", "Happiness21", "adasdasdfaf", "doujianghuimian", "2639450511967776"]
seed_id = ["JUGKING", "viper3", "MIDKING", "fatiaoahri", "ArkRuHa", "Rascal", "Tiancaishaonian", "Tar2an", "1zhangwukuaiqian", "tanoshiii"]


def requestsLog(url, status, headers):
    print(url)
    print(status)
    print(headers)


panth = ApiCaller(region, api_key, errorHandling=True, requestsLoggingFunction=requestsLog, debug=True)



async def getSummonerId(name):
    try:
        data = await panth.getSummonerByName(name)
        return data['puuid']
    except Exception as e:
        print(e)

async def getRecentMatchlist(accountId):
    try:
        data = await panth.getMatchList(accountId, params={"start": 0, "endIndex":10})
        return data
    except Exception as e:
        print(e)

async def getMatchRawData(matchId):
    try:
        data = await panth.getMatchMeta(matchId)
        return data
    except Exception as e:
        print(e)

'''
# rewrite this with new api
async def getRecentMatchlist(accountId):
    try:
        data = await panth.getMatchlist(accountId, params={"endIndex":10})
        return data
    except Exception as e:
        print(e)


async def getRecentMatches(accountId):
    try:
        matchlist = await getRecentMatchlist(accountId)
        tasks = [panth.getMatch(match['gameId']) for match in matchlist['matches']]
        return await asyncio.gather(*tasks)
    except Exception as e:
        print(e)
'''
def saveFiles(match, visited, seed):
    with open("data/matchIds.json", "w") as fout:
        json.dump(list(match), fout)
    with open("data/visited_id.json", "w") as fout:
        json.dump(list(visited), fout)
    with open("data/seeds.json", "w") as fout:
        json.dump(seed, fout)


# for python 3.7 and above
# puuid_seed = [asyncio.run(getSummonerId(name)) for name in seed_id]


# for python 3.6
loop = asyncio.get_event_loop()
puuid_seed = [loop.run_until_complete(getSummonerId(name)) for name in seed_id]
# loop.close()
###########################################

matches = set()
visited = set()
ts = time.time()
while puuid_seed and len(matches) < 10000:
    try:
        pl = puuid_seed.pop(0)

        # for python 3.7 and above
        # recent = asyncio.run(getRecentMatchlist(pl))

        # for python 3.6
        # loop = asyncio.get_event_loop()
        recent = loop.run_until_complete(getRecentMatchlist(pl))
        # loop.close()
        ###################################################

        matches.update(recent)
        visited.add(pl)
        for m in recent:
            # 
            # participants = asyncio.run(getMatchRawData(m))["metadata"]["participants"]

            # 
            # loop = asyncio.get_event_loop()
            participants = loop.run_until_complete(getMatchRawData(m))["metadata"]["participants"]
            # loop.close()

            for p in participants:
                if p not in visited and p not in puuid_seed:
                    puuid_seed.append(p)
        saveFiles(matches, visited, puuid_seed)
    except Exception as e:
        print(e)
        continue
# for python 3.6, end asyncio event loop
loop.close()


te = time.time() - ts
hh = te // 3600
mm = (te % 3600) // 60
ss = te % 60

print("Execution time: %d hours  %d minutes  %d seconds" % (hh, mm, ss))

# summonerId, accountId, puu = asyncio.run(getSummonerId(name))

# print("s : ", summonerId)
# print("a : ", accountId)
# print("p : ", puu) # this works.
# recent = asyncio.run(getRecentMatchlist(puu))
# print(recent)
# partic = asyncio.run(getMatchRawData(recent[1]))["metadata"]["participants"]
# print(partic)











