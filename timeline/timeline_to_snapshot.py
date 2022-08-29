# for each timeline, read and parse it to snapshots of the game

from riot_api import ApiCaller
import asyncio

import time, re, os, json, random
import numpy as np
import pandas as pd
from tqdm import tqdm


def requestsLog(url, status, headers):
    print(url)
    print(status)
    print(headers)

region = "kr"
api_key = "RGAPI-f6e08848-4ddf-4085-bacf-69a4d2e001f1"
panth = ApiCaller(region, api_key, errorHandling=True, requestsLoggingFunction=requestsLog, debug=True)


async def getTimeline(matchId):
    try:
        data = await panth.getTimeline(matchId)
        return data
    except Exception as e:
        print(e)
    # await asyncio.sleep(3) # fixed sleep time

async def getMatchRawData(matchId):
    try:
        data = await panth.getMatchMeta(matchId)
        return data
    except Exception as e:
        print(e)
    # await asyncio.sleep(3) # fixed sleep time

async def sleep(val):
    await asyncio.sleep(val)


def init_champ_status(): # use a ndarray to record
    result = np.zeros((6, 56)) # 6 item slots, 56d features


def pprint(itemslot, check):
    for k in range(1, 11):
        v = itemslot[k]
        rubbish = [check[str(id)]["name"] if id!=0 else 0 for id in v]
        print(rubbish)


def convert_to_full_list(record):
    # convert the ndarray in recording list so that it can be saved as json
    for idx, ar in enumerate(record):
        if isinstance(ar, list):
            convert_to_full_list(ar)
        elif isinstance(ar, np.ndarray):
            record[idx] = list(ar)


with open("data/matchIds.json") as fin:
    matchIds = json.load(fin)
random.shuffle(matchIds)
with open("item.json") as fin:
    items = json.load(fin)
all_items = items["data"] # dict: {item_id: {name:, description:, colloq:, ....}, }
with open("champion.json", encoding="utf-8") as fin:
    check_champions = json.load(fin)
    check_champions = check_champions["data"]
    check_champions = {k.lower(): v["name"].replace(" ", "_").lower() for k, v in check_champions.items()}

champion_root = "../src/champ/en_nv/"
item_root = "../src/item/en/"

loop = asyncio.get_event_loop()
for mid in matchIds[:1000]:
    SAVE_SNAPSHOTS = []

    try:
        mmmm = loop.run_until_complete(getMatchRawData(mid))
        tttt = loop.run_until_complete(getTimeline(mid)) # json file
    except Exception as e:
        print(e)
        loop.run_until_complete(sleep(3))
        continue
    # meta = tttt["meta"] # player, champion, position
    # 要用https://developer.riotgames.com/apis#match-v5/GET_getMatch这个api去确定每个人都用的什么英雄
    # champion name的部分需要小改 # 已改
    champs = [partic["championName"] for partic in mmmm["info"]["participants"]]

    # champs = [v[0] for _, v in meta.items()] # 10 participating champions
    # for every champ, use an ndarray to record its info
    # drop first 2 columns: idx, ability_id
    profiles = [pd.read_csv(champion_root + check_champions[ch.lower()] + ".csv").values[:, 2:] for ch in champs] 

    record_itemslot = {idx: [0, 0, 0, 0, 0, 0, 0] for idx in range(1, 11)} # 6 item slots, and 1 trinket slot
    record_skillslot = {idx: [0, 0, 0, 0] for idx in range(1, 11)}

    # initial champion status with 6 item slots and an innate passive ability <- add 1 trinket slot
    status = [np.zeros((7, 56)) for _ in range(10)] # for all 10 champs, each has 6 item slots, 1 trinket slot, feature dim is 56
    for idx, pf in enumerate(profiles):
        passive = pf[0, :]
        status[idx] = np.row_stack((status[idx], passive))

    # participantsId: 1 ~ 10
    for idx, frame in enumerate(tttt["info"]["frames"]):
        # frame: dict_keys(['events', 'participantFrames', 'timestamp'])
        for event in frame["events"]:
            time_stamp = event["timestamp"]
            if event["type"] == "ITEM_PURCHASED":
                itemId = event["itemId"]
                participantId = event["participantId"]

                item_info = all_items[str(itemId)]
                name = item_info["name"]
                
                path = item_root + name.replace(" ", "_").lower() + ".csv"
                if not os.path.exists(path):
                    continue
                item_val = pd.read_csv(path).values[0, 2:]

                if "from" in item_info.keys():
                    for ff in item_info["from"]:
                        if ff in record_itemslot[participantId]:
                            iiii = record_itemslot[participantId].index(ff)
                            status[participantId - 1][iiii] = 0
                            record_itemslot[participantId] = 0
                iiii = record_itemslot[participantId].index(0)
                status[participantId - 1][iiii] = item_val
                record_itemslot[participantId][iiii] = itemId

            elif event["type"] == "ITEM_UNDO":
                participantId = event["participantId"]
                beforeId = event["beforeId"]
                afterId = event["afterId"]

                if beforeId in record_itemslot[participantId]:
                    iiii = record_itemslot[participantId].index(beforeId)
                    status[participantId - 1][iiii] = 0
                    record_itemslot[participantId][iiii] = 0
                if afterId == 0:
                    continue

                item_info = all_items[str(afterId)]
                name = item_info["name"]

                path = item_root + name.replace(" ", "_").lower() + ".csv"
                if not os.path.exists(path):
                    continue
                item_val = pd.read_csv(path).values[0, 2:]

                iiii = record_itemslot[participantId].index(0)
                status[participantId - 1][iiii] = item_val
                record_itemslot[participantId][iiii] = afterId
            
            elif event["type"] == "ITEM_DESTROYED" or event["type"] == "ITEM_SOLD":
                participantId = event["participantId"]
                itemId = event["itemId"]

                if itemId in record_itemslot[participantId]:
                    iiii = record_itemslot[participantId].index(itemId)
                    record_itemslot[participantId][iiii] = 0
                    status[participantId - 1][iiii] = 0
            
            elif event["type"] == "WARD_PLACED":
                pass

            elif event["type"] == "SKILL_LEVEL_UP":
                participantId = event["participantId"]
                skill_slot = event["skillSlot"]

                n_rows = profiles[participantId - 1].shape[0]
                if n_rows == 5:
                    status[participantId - 1] = np.row_stack((status[participantId - 1], profiles[participantId - 1][skill_slot, :]))
                
                elif n_rows == 8:
                    if skill_slot == 4:
                        status[participantId - 1] = np.row_stack((status[participantId - 1], profiles[participantId - 1][skill_slot, :]))
                    else:
                        status[participantId - 1] = np.row_stack((status[participantId - 1], profiles[participantId - 1][skill_slot*2-1: skill_slot*2+1, :]))

                elif n_rows == 9:
                    if skill_slot == 4:
                        status[participantId - 1] = np.row_stack((status[participantId - 1], profiles[participantId - 1][skill_slot, :]))
                    else:
                        status[participantId - 1] = np.row_stack((status[participantId - 1], profiles[participantId - 1][skill_slot*2: skill_slot*2+2, :]))

                else:
                    print(champs[participantId - 1], id, time_stamp)
                    continue

            elif event["type"] == "GAME_END":
                winner = event["winningTeam"]
                # winner == 100 -> team 1 wins
                # winner == 200 -> team 2 wins
            # save snapshot
        temp = [ar.tolist() for ar in status]
        SAVE_SNAPSHOTS.append(temp)
        # time.sleep(1) # cannot use time.sleep -> this will cause main thread get stuck.
    with open("data/snapshot/" + str(mid) + ".json", 'w') as fout:
        # SAVE_SNAPSHOTS = convert_to_full_list(SAVE_SNAPSHOTS)
        json.dump((winner, SAVE_SNAPSHOTS), fout)

loop.close()










