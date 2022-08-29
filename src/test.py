from cgitb import text
import os, json, re, time
from tarfile import PAX_NAME_FIELDS
from shutil import register_unpack_format
from tkinter.font import names
from urllib.request import urlopen
from bs4 import BeautifulSoup as BS
from tqdm import tqdm

import pandas as pd


# target = "https://leagueoflegends.fandom.com/wiki/Miss_Fortune/LoL"
items = "https://leagueoflegends.fandom.com/wiki/Item_(League_of_Legends)"
wiki_root = "https://leagueoflegends.fandom.com"




page = urlopen(url=items).read()
soup = BS(page, "html.parser", from_encoding="utf-8")
# links = list(map(get_url, heros))


def clean(text):
    text = text.replace("\n", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("%27", "'")
    text = text.replace("%26", "&")
    return text


def get_item_urls(soup):
    soup = soup.find_all("div", class_="tlist")[:10] # drop removed items
    # print(len(soup))
    # divs = soup.find_all("div", class_="item-icon tooltips-init-complete")
    # print(len(divs))
    links = []
    for div in soup:
        links += div.find_all("a", href=re.compile(r"/wiki/"))

    # links = soup[9].find_all("a", href=re.compile(r"/wiki/"))
    links = [a["href"] for a in links]
    return links


def fetch_one_item(item_url):
    page = urlopen(url=item_url).read()
    soup = BS(page, "html.parser", from_encoding="utf-8")
    
    dddd = soup.find_all("section", class_="pi-item pi-group pi-border-color")
    dddd = [box.text for box in dddd]
    text = clean(" ".join(dddd))
    # print(dddd)

    record = [0] * 9
    if "critical" in text:
        record[0] = 1
    if "attack damage" in text:
        record[1] = 1
    if "ability power" in text:
        record[2] = 1
    if "armor" in text:
        record[3] = 1
    if "magic resistance" in text:
        record[4] = 1
    if "on-hit" in text:
        record[5] = 1
    if "lethality" in text:
        record[6] = 1
    if "armor penetration" in text:
        record[7] = 1
    if "magic penetration" in text:
        record[8] = 1
    
    return record


def get_champ_one_ability(div):
    record = [0] * 8
    
    # print("=================================================")
    for ab in div:
        text = ab.text
        # print(text, "!!!!!!!!!")
        text = text.split("COOLDOWN:")[-1]
        text = " ".join(text.split()).lower()
        
        if "cooldown" in text and "reduce" in text:
            record[0] = 1
        if "active" in text:
            record[1] = 1
        if "passive" in text:
            record[2] = 1
        if "% ad" in text or "bonus ad" in text:
            record[3] = 1
        if "% ap" in text or "bonus ap" in text:
            record[4] = 1
        if "armor" in text or "magic resistance" in text or "maximum health" in text:
            record[5] = 1
        if "target's maximum health" in text or "target's current health" in text or "target's missing health" in text:
            record[6] = 1
    # print("=================================================")
    return record


def nv_get_one_ability(div):
    record = [0] * 9 # [crit_chance, bonus_ad, bonus_ap, bonus_armor, bonus_resistance, on-hit, lethality, ad_pen, ap_pen]
    
    for ab in div:
        text = ab.text
        text = text.split("COOLDOWN:")[-1]
        text = " ".join(text.split()).lower()
        
        if "critical" in text:
            record[0] = 1
        if "attack damage" in text:
            record[1] = 1
        if "ability power" in text:
            record[2] = 1
        if "armor" in text:
            record[3] = 1
        if "magic resistance" in text:
            record[4] = 1
        if "on-hit" in text:
            record[5] = 1
        if "lethality" in text:
            record[6] = 1
        if "armor penetration" in text:
            record[7] = 1
        if "magic penetration" in text:
            record[8] = 1
    return record


def get_champ_all_ability(champ_url):
    record = dict()
    
    abbs = ["skill skill_innate", "skill skill_q", "skill skill_w", "skill skill_e", "skill skill_r"] # div
    champ_p = urlopen(url=champ_url).read()
    champ_s = BS(champ_p, "html.parser", from_encoding="utf-8")
    
    # tmp = champ_s.find_all("div", class_=abbs[1])
    # print(len(tmp))
    
    # tt = tmp[0].text
    # print(" ".join(tt.split()))
    for abid in abbs:
        # print(abid)
        tmp = champ_s.find_all("div", class_=abid)
        # val = get_champ_one_ability(tmp)
        val = nv_get_one_ability(tmp)
        key = abid.split("_")[1]
        
        record[key] = val
    return record



in_root = "src/item/"
out_root = "src/item/en/"


urls = get_item_urls(soup)
item_names = [clean(url[6:]).lower() + ".csv" for url in urls]
urls = [wiki_root + url for url in urls]


new_cols = ["critical_strike", "bonus_ad", "bonus_ap", "bonus_armor", "bonus_resistance", "on-hit", "lethality", "ad_pen", "ap_pen"]

for idx in range(len(item_names)):
    name = item_names[idx]
    url  = urls[idx]
    if os.path.exists(in_root + name):
        df = pd.read_csv(in_root + name)
        features = fetch_one_item(url)

        for col, val in zip(new_cols, features):
            df[col] = val
        
        df = df.drop(columns = "Unnamed: 0")
        df.to_csv(out_root + name)
    else:
        print(name)

# fetch_one_item("https://leagueoflegends.fandom.com/wiki/Sunfire_Aegis")