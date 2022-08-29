import asyncio
from xmlrpc.server import SimpleXMLRPCRequestHandler
import aiohttp
import json
from functools import wraps

from utils import utils as utils
from utils import exceptions as exc

from RateLimit.RateLimiterManager import RateLimiterManager

class ApiCaller():
    
    BASE_URL = "https://{server}.api.riotgames.com/"
    BASE_URL_LOL = BASE_URL + "lol/"
    BASE_URL_TFT = BASE_URL + "tft/"
    BASE_URL_LOR = BASE_URL + "lor/"
    BASE_URL_RIOT = BASE_URL + "riot/"
    BASE_URL_VAL = BASE_URL + "val/"
    
    # PLATFORMS = ["br1","eun1","euw1","jp1","kr","la1","la2","na1","oc1","tr1","ru"]
    # self.REGIONS = {"kr": "asia", "jp1": "asia", "eun1": "europe", "euw1": "europe", "ru": "europe", "na1": "americas", "la1": "americas", "la2": "americas"}
    # ["americas","asia","europe"]
    # TOURNAMENT_REGIONS = ["americas"]
    
    
    def __init__(self, server, api_key, errorHandling = False, requestsLoggingFunction = None, debug=False):
        """
        Initialize an instance of Pantheon class
        
        :param string server: The server Pantheon will target for the requests. An instance is intended to only call one server. Use multiple instances of Pantheon to call multiples servers.
        It can take the values described there : https://developer.riotgames.com/regional-endpoints.html (euw1, na1...)
        :param string api_key: The API key needed to call the Riot API
        :param boolean errorHandling: Precise if Pantheon should autoretry after a ratelimit (429) or server error (5XX). Default is False
        :param boolean debug: Allows to print debug messages. Default is False
        """
        self.REGIONS = {"kr": "asia", "jp1": "asia", "eun1": "europe", "euw1": "europe", "ru": "europe", "na1": "americas", "la1": "americas", "la2": "americas"}

        self._key = api_key
        self._server = server
        self._region = self.REGIONS[server]
        self._rl = RateLimiterManager(debug)
        
        self.errorHandling = errorHandling
        self.requestsLoggingFunction = requestsLoggingFunction
        self.debug = debug
        
    def __str__(self):
        return str(self._rl)
    
    def locked(self):
        """
        Return True if at least one limiter is locked
        """
        return self._rl.locked()

    def ratelimit(func):
        """
        Decorator for rate limiting.
        It will handle the operations needed by the RateLimiterManager to assure the rate limiting and the change of limits considering the returned header.
        """
        @wraps(func)
        async def waitLimit(*args, **params):
            token = await args[0]._rl.getToken(func.__name__)
            
            response = await func(*args, **params)
            
            try:
                limits = utils.getLimits(response.headers)
                timestamp = utils.getTimestamp(response.headers)
            except:
                limits = None
                timestamp = utils.getTimestamp(None)
            
            await args[0]._rl.getBack(func.__name__, token, timestamp, limits)
            
            return response
            
        return waitLimit
    
    def errorHandler(func):
        """
        Decorator for handling some errors and retrying if needed.
        """
        @wraps(func)
        async def _errorHandling(*args, **params):
            """
            Error handling function for decorator
            """
            if not args[0].errorHandling:
                return await func(*args, **params)
            else:
                try:
                    return await func(*args, **params)
                #Errors that should be retried
                except exc.RateLimit as e:
                    if args[0].debug:
                        print(e)
                        print("Retrying")
                    i = e.waitFor()
                    while i < 6:
                        await asyncio.sleep(i)
                        try:
                            return await func(*args, **params)
                        except Exception as e2:
                            if args[0].debug:
                                print(e2)
                        i += 2
                    raise e
                except (exc.ServerError, exc.Timeout) as e:
                    if args[0].debug:
                        print(e)
                        print("Retrying")
                    i = 1
                    while i < 6:
                        await asyncio.sleep(i)
                        try:
                            return await func(*args, **params)
                        except (exc.Timeout, exc.ServerError) as e2:
                    
                            pass
                        i += 2
                        if args[0].debug:
                            print(e2)
                            print("Retrying")
                    print("there is no bug")
                    raise e
                except (exc.NotFound, exc.BadRequest) as e:
                    raise e
                except (exc.Forbidden, exc.Unauthorized,) as e:
                    print(e)
                    raise SystemExit(0)
                except Exception as e:
                    raise e
                
        return _errorHandling
    
    def exceptions(func):
        """
        Decorator translating status code into exceptions
        """
        @wraps(func)
        async def _exceptions(*args, **params):
            
            response = await func(*args, **params)
            
            if response is None:
                raise exc.Timeout
            
            elif response.status == 200:
                return json.loads(await response.text())

            elif response.status == 404:
                raise exc.NotFound
                
            elif response.status in [500,502,503,504]:
                raise exc.ServerError
                
            elif response.status == 429:
                raise exc.RateLimit(response.headers)
                
            elif response.status == 403:
                raise exc.Forbidden
                
            elif response.status == 401:
                raise exc.Unauthorized
                
            elif response.status == 400:
                raise exc.BadRequest
                
            elif response.status == 408:
                raise exc.Timeout
                
            else:
                raise Exception("Unidentified error code : "+str(response.status))
            
        return _exceptions
        
    async def fetch(self, url, method="GET", data=None):
        """
        Returns the result of the request of the url given in parameter after attaching the api_key to the header
        """
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "X-Riot-Token": self._key
            }
            
            try:
                if method=="GET":
                    response = await session.request("GET", url, headers=headers)
                else:
                    response = await session.request(method, url, headers=headers, data=json.dumps(data))
            #In case of timeout
            except Exception as e:
                return None
            
            #If a logging function is passed, send it url, status code and headers
            if self.requestsLoggingFunction:
                self.requestsLoggingFunction(url, response.status, response.headers)
            
            #await response.text() needed here in the client session, dunno why
            await response.text()
            return response
    
    @errorHandler
    @exceptions
    @ratelimit
    async def getSummonerByName(self, summonerName):
        """
        :param string summonerName: name of the player
        
        Returns the result of https://developer.riotgames.com/api-methods/#summoner-v4/GET_getBySummonerName
        """
        return await self.fetch((self.BASE_URL_LOL + "summoner/v4/summoners/by-name/{summonerName}").format(server=self._server, summonerName=summonerName))
   
    @errorHandler
    @exceptions
    @ratelimit
    async def getMatchList(self, puuid, params=None):
        """
        :param string puuid: unique id of the player
        
        Returns the result of https://developer.riotgames.com/apis#match-v5/GET_getMatchIdsByPUUID
        """
        return await self.fetch((self.BASE_URL_LOL + "match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={end}").format(server=self._region, puuid=puuid, start=params["start"], end=params["endIndex"]))
   
    @errorHandler
    @exceptions
    # @ratelimit
    async def getMatchMeta(self, matchId):
        """
        :param string matchId: matchid
        
        Returns the result of https://developer.riotgames.com/apis#match-v5/GET_getMatch
        """
        return await self.fetch((self.BASE_URL_LOL + "match/v5/matches/{matchid}").format(server=self._region, matchid=matchId))

    @errorHandler
    @exceptions
    # @ratelimit
    async def getTimeline(self, matchId):
        """
        :param string MatchId
        
        Returns the result of https://developer.riotgames.com/apis#match-v5/GET_getTimeline
        """
        return await self.fetch((self.BASE_URL_LOL + "match/v5/matches/{matchid}/timeline").format(server=self._region, matchid=matchId))
   




