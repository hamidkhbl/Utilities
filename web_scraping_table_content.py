#%%
import requests
import os
import pandas as pd
import bs4 as bs 
import time

url = 'https://www.health.gov.au/news/health-alerts/novel-coronavirus-2019-ncov-health-alert/coronavirus-covid-19-current-situation-and-case-numbers'

folder_name = 'html_source'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

response = requests.get(url)
with open(folder_name+'/source.html', mode = 'wb') as file:
    file.write(response.content)

with open(folder_name+'/source.html') as file:
    soup = bs.BeautifulSoup(file, 'lxml')

table = soup.find('table')
table_header = table.find_all('th')
table_rows = table.find_all('tr')

all_rows = [['Location','Confirmed cases']]
for tr in table_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    all_rows.append(row)
header = all_rows.pop(0)
df = pd.DataFrame(all_rows, columns=header)
df = df.replace('\n','', regex=True)
df = df[df['Confirmed cases'].notnull()]
print(df)
df.to_csv('data.csv')
# %%
