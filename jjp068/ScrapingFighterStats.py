import requests
import re
from bs4 import BeautifulSoup as bs
import pandas as pd
from string import ascii_lowercase

"""
References: 

Title: pandas
Author: pandas Team
Availability: https://github.com/pandas-dev/pandas
Version:  1.2.4

Title: BeautifulSoup4
Author: BeautifulSoup Team
Availability: https://github.com/akalongman/python-beautifulsoup/blob/master/LICENSE
Version:  4.9.3


"""

fighterlist = []
for c in ascii_lowercase:
    #loads list of ufc fighter
    r2 = requests.get('http://ufcstats.com/statistics/fighters?char={}&page=all'.format(c))
    # convert to a beautiful soup
    fighter_list_page = bs(r2.content, features="lxml")

    links = [str(i['href']) for i in fighter_list_page.find_all('a',attrs= {'class':'b-link b-link_style_black'}, href = True, text=True)]
    linksset = set(links)
    print(linksset)



    for l in linksset:
        # loads ufc stat content
        r = requests.get(l)
        # convert to a beautiful soup
        fighter_page = bs(r.content, features="lxml")

        #column names
        #Name and Record
        nme_rec_headers =['Name','Record']
        #get physical traits
        traits = fighter_page.findAll("i", attrs= {'class':"b-list__box-item-title b-list__box-item-title_type_width"})
        trait_headers = [str(t.get_text().strip()[:-1]) for t in traits[:-1]]
        #get career averages
        stats = fighter_page.findAll("i", attrs = {"class": "b-list__box-item-title b-list__box-item-title_font_lowercase b-list__box-item-title_type_width"})
        stat_headers = [str(s.get_text().strip()[:-1]) for s in stats]
        #merged total column headers
        column_headers = nme_rec_headers+ trait_headers+ stat_headers


        #fighter information
        # name & recod
        name_item = fighter_page.find('span', attrs= {'class':"b-content__title-highlight"})
        name_text = name_item.get_text().strip()
        record_item = fighter_page.find('span',attrs= {'class':"b-content__title-record"})
        record_text = record_item.get_text().strip()
        record_formatted = re.sub("Record:","",record_text)
        name_rec_info = [name_text,record_formatted]

        # Stat info
        info1 = fighter_page.findAll('li', attrs= {'class' :"b-list__box-list-item b-list__box-list-item_type_block"})
        proper_info1 = [re.sub("Height:|Weight:|DOB:|Reach:|lbs\.|STANCE:|SLpM:|Str. Acc.:|%|SApM:|Str. Def:|TD Avg.:|TD Acc.:|TD Def.:|Sub. Avg.:","",str(i.get_text()))for i in info1]
        proper_info2 = [str(i).strip() for i in proper_info1]
        del proper_info2[9]
        all_info = name_rec_info + proper_info2
        fighterlist.append(all_info)

fighter_frame = pd.DataFrame(fighterlist,columns=column_headers)
print(fighter_frame)
#fighter_frame.to_csv('practiceallfighterinfo.csv')






