# data preparation for display

from riot_api import ApiCaller
import asyncio


import time, re, json
import numpy as np
from tqdm import tqdm

test_MatchID = "KR_5702061589"

region = "kr"
api_key = ""
# api_key = "RGAPI-65a944e8-6beb-43c0-b063-236d15051a31"

def requestsLog(url, status, headers):
    print(url)
    print(status)
    print(headers)


panth = ApiCaller(region, api_key, errorHandling=True, requestsLoggingFunction=requestsLog, debug=True)

async def getTimeline(matchId):
    try:
        data = await panth.getTimeline(matchId)
        return data
    except Exception as e:
        print(e)

async def getMatchRawData(matchId):
    try:
        data = await panth.getMatchMeta(matchId)
        return data
    except Exception as e:
        print(e)

# ==================================================

def getPos(timeline):
    # timeline: dict
    # keys: ['frameInterval', 'frames': list[dict], 'gameId', 'participants']
    pos = []
    for frame in timeline["frames"]:
        if "participantFrames" not in frame:
            continue
        tmp = []
        for ch in range(1, 11):
            x = frame["participantFrames"][str(ch)]["position"]["x"]
            y = frame["participantFrames"][str(ch)]["position"]["y"]
            tmp.append((x, y))
        pos.append(tmp)
    return pos



participant_champion = dict() # puuid: (champ_name, position)

# coro = [getMatchRawData(test_MatchID)]
loop = asyncio.get_event_loop()
# data = loop.run_until_complete(asyncio.wait(coro))
match = loop.run_until_complete(getMatchRawData(test_MatchID))
# loop.close()

for partic in match["info"]["participants"]:
    # lane: TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
    participant_champion[partic["puuid"]] = (partic["championName"], partic["teamPosition"])
print(participant_champion)

# for python 3.7 and later
# timeline = asyncio.run(getTimeline(test_MatchID))
# print(type(timeline["info"]["frames"][0]))

# for python 3.6
# loop = asyncio.get_event_loop()
timeline = loop.run_until_complete(getTimeline(test_MatchID))
loop.close()

match_data = {"meta": participant_champion, "timeline": timeline}

with open("data/test_timeline.json", 'w') as fout:
   json.dump(match_data, fout)
print("\n\n\nDone.")




