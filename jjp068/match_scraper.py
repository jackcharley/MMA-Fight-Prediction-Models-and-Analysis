import requests
import re
from bs4 import BeautifulSoup as bs
import pandas as pd

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
#loads list of ufc events

r2 = requests.get('http://ufcstats.com/statistics/events/completed?page=all')

#convert to beautiful soup

event_list_soup = bs(r2.content,features='lxml')

# links to each event

links = [str(i['href']) for i in event_list_soup.find_all('a',attrs= {'class':'b-link b-link_style_black'}, href = True, text=True)]



#empty list to add all fight stats
l =[]

# loads event information

r = requests.get('http://ufcstats.com/event-details/9ddfb3369a3b0394')

# convert to beautiful soup

event_page = bs(r.content,features= 'lxml')

#loads list of ufc events

r2 = requests.get('http://ufcstats.com/statistics/events/completed?page=all')

#convert to beautiful soup

event_list_soup = bs(r2.content,features='lxml')

# links to each event

links = [str(i['href']) for i in event_list_soup.find_all('a',attrs= {'class':'b-link b-link_style_black'}, href = True, text=True)]

# event info column headers

event_column_info = [str(e.get_text().strip()) for e in event_page.find_all("th",attrs= { 'class':'b-fight-details__table-col'})]
event_column_info.insert(0,'W/L')
event_column_info [1]= 'Fighter1'
event_column_info.insert(2,'Fighter2')
event_column_info[3] = 'F1_Kd'
event_column_info.insert(4,'F2_Kd')
event_column_info[5] = 'F1_Str'
event_column_info.insert(6,'F2_Str')
event_column_info[7] = 'F1_Td'
event_column_info.insert(8,'F2_Td')
event_column_info[9] = 'F1_Sub'
event_column_info.insert(10,'F2_Sub')
event_column_info.remove('Sub')
print(event_column_info)

for link in links:
    # loads event information

    r = requests.get(link)

    # convert to beautiful soup

    event_page = bs(r.content, features='lxml')

    # event information

    table_row = event_page.find("tbody").find_all('tr')
    for tr in table_row:
        td = tr.find_all("td")
        row = [str(tr.get_text().strip())for tr in td]
        formatted_row = [info for segments in row for info in segments.splitlines()]
        final_info = [row for row in formatted_row if row.strip()]
        final_info = [info.strip() for info in final_info]
        if final_info[1]== 'nc'or final_info[1]== 'draw':
            del final_info[1]
        if  final_info[13] not in ('1','2','3','4','5'):
            del final_info[13]
        print(final_info)
        l.append(final_info)

match_frame = pd.DataFrame(data=l,columns=event_column_info)
print(match_frame)
match_frame.to_csv("newfightinfo.csv")




