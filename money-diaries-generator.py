from bs4 import BeautifulSoup
import requests
import markov_novel
import markovify
from pprint import pprint

general_url = 'https://www.refinery29.com/en-us/money-diary?page='
all_urls = []

def get_all_urls(general_url, all_urls, n):
    for i in range(1, n+1):
        page_url = general_url + str(i)
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_divs = soup.find_all('div', ['card prime', 'card featured-story', 'card standard'])
        for div in page_divs:
            if '/en-us/money-diary-' == div.a.get('href')[0:19]:
                all_urls.append('https://www.refinery29.com/' + div.a.get('href'))

def get_single_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    page_divs = soup.find_all("div", class_='section-text')
    page_text = ''
    for item in page_divs:
        page_text += item.get_text()
    return page_text


def main():
  # get_all_urls(general_url, all_urls, 31)
  corpus = ''
  get_all_urls(general_url, all_urls, 3)
  for url in all_urls:
    corpus += get_single_url(url)
  text_model = markovify.Text(corpus)
  novel = markov_novel.Novel(text_model, chapter_count=20)
  novel.write(novel_title='refinery29-generated-money-diary', filetype='txt')

if __name__== "__main__":
  main()
