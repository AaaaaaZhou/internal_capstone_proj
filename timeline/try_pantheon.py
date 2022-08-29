from pantheon import pantheon
import asyncio

import time, re, json



region  = "kr"
api_key = "RGAPI-1ef63024-5d27-44f2-a443-815d1c43a778"


def requestsLog(url, status, headers):
    print(url)
    print(status)
    print(headers)


panth = pantheon.Pantheon(region, api_key, errorHandling=True, requestsLoggingFunction=requestsLog, debug=True)



async def getSummonerId(name):
    try:
        data = await panth.getSummonerByName(name)
        return data['id'],data['accountId'], data['puuid']
    except Exception as e:
        print(e)

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




name = "fragiIe"

summonerId, accountId, puu = asyncio.run(getSummonerId(name))

print("s : ", summonerId)
print("a : ", accountId)
print("p : ", puu) # this works.
print(asyncio.run(getRecentMatchlist(accountId)))





