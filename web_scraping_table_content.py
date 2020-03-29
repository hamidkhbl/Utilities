
# Australia's government updates data related to Covid-19 several times in a day
# This code extracts the data from www.health.gov.au and saves it in a CSV file for analysis purposes.
#%%
# imports
import requests
import os
import pandas as pd
import bs4 as bs 
import datetime

# save the response in a html file
url = 'https://www.health.gov.au/news/health-alerts/novel-coronavirus-2019-ncov-health-alert/coronavirus-covid-19-current-situation-and-case-numbers'

html_folder_name = 'html_source'
if not os.path.exists(html_folder_name):
    os.makedirs(html_folder_name)

csv_folder_name = 'csv'
if not os.path.exists(csv_folder_name):
    os.makedirs(csv_folder_name)

response = requests.get(url)
with open(html_folder_name+'/source.html', mode = 'wb') as file:
    file.write(response.content)

# find table from the html
with open(html_folder_name+'/source.html') as file:
    soup = bs.BeautifulSoup(file, 'lxml')

table = soup.find('table')
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

date = str(datetime.datetime.now()).replace(':','_')
# compare with the last data, save if it is changed
if os.path.exists(csv_folder_name+'/data.csv'):
    df_old = pd.read_csv(csv_folder_name+'/data.csv', index_col=0)
    diff = (df != df_old).stack()
    changed = diff[diff]
    if changed.shape[0] > 0:
        # save the data in a csv file
        df.to_csv(csv_folder_name+'/data.csv')
        df.to_csv(csv_folder_name+'/data_'+date+'.csv')
else:
    df.to_csv(csv_folder_name+'/data.csv')
    df.to_csv(csv_folder_name+'/data_'+date+'.csv')

# %%
