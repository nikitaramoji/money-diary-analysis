from bs4 import BeautifulSoup
import requests
from pprint import pprint

url = 'https://www.refinery29.com/en-us/money-diary?page='
divs = []
for i in range(1, 32):
    page_url = url + str(i)
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    page_divs = soup.findAll('div', ['card prime', 'card featured-story', 'card standard'])
    for div in page_divs:
        divs.append(div.a.get('href'))
pprint(divs)
print(len(divs))
