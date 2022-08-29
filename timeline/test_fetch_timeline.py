from riot_api import ApiCaller
import asyncio

import time, re, os, json
import numpy as np
import pandas as pd
from tqdm import tqdm


def requestsLog(url, status, headers):
    print(url)
    print(status)
    print(headers)

region = "kr"
api_key = ""
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


def init_champ_status(): # use a ndarray to record
    result = np.zeros((6, 56)) # 6 item slots, 56d features


def pprint(itemslot, check):
    for k in range(1, 11):
        v = itemslot[k]
        rubbish = [check[str(id)]["name"] if id!=0 else 0 for id in v]
        print(rubbish)


with open("data/matchIds.json") as fin:
    matchIds = json.load(fin)
with open("item.json") as fin:
    items = json.load(fin)


loop = asyncio.get_event_loop()

id = matchIds[0]
tttt = loop.run_until_complete(getTimeline(id)) # json file
print(tttt.keys())

print(tttt["metadata"].keys())
print(tttt["metadata"])

print("================================")

print(tttt["info"].keys())
print(tttt["info"]["frames"])

loop.close()






