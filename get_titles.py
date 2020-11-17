import requests
import pandas as pd
from bs4 import BeautifulSoup

URL = ['https://pubmed.ncbi.nlm.nih.gov/collections/60202498/?sort=pubdate',
'https://pubmed.ncbi.nlm.nih.gov/collections/60202498/?sort=pubdate&page=2',
'https://pubmed.ncbi.nlm.nih.gov/collections/60202498/?sort=pubdate&page=3',
'https://pubmed.ncbi.nlm.nih.gov/collections/60202498/?sort=pubdate&page=4',
'https://pubmed.ncbi.nlm.nih.gov/collections/60202498/?sort=pubdate&page=5',
'https://pubmed.ncbi.nlm.nih.gov/collections/60202498/?sort=pubdate&page=6',
'https://pubmed.ncbi.nlm.nih.gov/collections/60202498/?sort=pubdate&page=7',
'https://pubmed.ncbi.nlm.nih.gov/collections/60202498/?sort=pubdate&page=8',
'https://pubmed.ncbi.nlm.nih.gov/collections/60202498/?sort=pubdate&page=9',
'https://pubmed.ncbi.nlm.nih.gov/collections/60202498/?sort=pubdate&page=10']

row = []
titles = []
href = []
abstracts = []
years = []
authors = []


for u in URL:
    #get titles and hrefs
    page = requests.get(u)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.findAll(class_='docsum-title')

    for i in results:
        row.append(i.attrs['data-ga-action'])
        titles.append(i.get_text().strip())
        href.append(i.attrs['href'])

        # get abstract
        abs_page = requests.get("https://pubmed.ncbi.nlm.nih.gov"+i.attrs['href'])
        soup2 = BeautifulSoup(abs_page.content, 'html.parser')
        abstract = soup2.find(class_='abstract')
        text = abstract.get_text()
        if text.strip() == 'Abstract':
            abstract = soup2.find(id='enc-abstract')
            text = abstract.get_text()
        abstracts.append(text.strip())
        #print(text.strip())
        years.append(soup2.find('span', {'class' : 'cit'}).get_text().strip())
        #authors.append(soup2.find(class_='full-name').get_text().strip())
        #authors-list
        authors_list = soup2.find(class_='authors-list')
        #print(authors_list)
        spans = authors_list.find_all("span", {"class": "authors-list-item"})
        #print(spans)
        print('------------------------------------------------------------')
        link = ''
        for span in spans:
            link += span.find(class_='full-name').get_text() + ' | '
        print(link)
        authors.append(link)




df = pd.DataFrame({'row':row, 'Title':titles, 'Author': authors,'Year':years, 'abstract':abstracts, 'PMID':href})
df.to_csv('export_2.csv')