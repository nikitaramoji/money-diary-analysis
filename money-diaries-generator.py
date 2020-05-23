from bs4 import BeautifulSoup
import requests
import markov_novel
import markovify
import pickle
from pprint import pprint

general_url = 'https://www.refinery29.com/en-us/money-diary?page='
corpus_url = 'corpus.txt'
all_urls = []

def get_all_urls(general_url):
    print("Getting all URLs")
    continue_boolean = True
    i = 1
    while(continue_boolean):
        print("On page " + str(i))
        page_url = general_url + str(i)
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_divs = soup.find_all('div', ['card prime', 'card featured-story', 'card standard'])
        if not page_divs:
            continue_boolean = False
        else:
            i += 1
            for div in page_divs:
                if '/en-us/money-diary-' == div.a.get('href')[0:19]:
                    all_urls.append('https://www.refinery29.com/' + div.a.get('href'))
    return all_urls

def get_single_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    page_divs = soup.find_all("div", class_='section-text')
    page_text = ''
    for item in page_divs:
        page_text += item.get_text()
    return page_text

def generate_corpus(general_url):
    all_urls = get_all_urls(general_url)
    corpus = []
    for url in all_urls:
        corpus.append(get_single_url(url))
    f = open('corpus.txt', 'wb')
    pickle.dump(corpus, f)
    f.close()

def generate_markovify_money_diary(corpus_url):
    f = open(corpus_url, 'rb')
    text = f.read().decode("utf-8")
    f.close()
    print("Generating text model")
    text_model = markovify.Text(text)
    print("Generating Markov Novel")
    novel = markov_novel.Novel(text_model, chapter_count=3)
    print("Writing to file")
    novel.write(novel_title='index', filetype='md')
    print("Done!")


def main():
  # generate_corpus(general_url)
  generate_markovify_money_diary(corpus_url)

if __name__== "__main__":
  main()
