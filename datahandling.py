import pandas as pd
import numpy as np
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import *
from wordcloud import WordCloud,STOPWORDS
import re
import datetime 
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

commodity_prices = pd.read_excel("External_Data.xls", header=0)
commodity_prices['Year_month'] = pd.to_datetime(commodity_prices['Year_month'], format='%YM%m')
commodity_prices.columns = ['Date', 'Price (USD)', 'Status']

# Getting rid of the day in the date to ease merge
commodity_prices['Date'] = commodity_prices['Date'].map(lambda x: x.strftime('%Y-%m'))

news_data = pd.read_csv("Combined_News_DJIA.csv", header=0)

# Read the date column using the correct format. Since the year is not padded by the century, make use of %y instead of %Y
news_data['Date'] = pd.to_datetime(news_data['Date'], format='%d/%m/%y')


# Merge all top news article for a date
news_data['combined_news']=news_data.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)

# Create a datetime index to group using

news_data.index = news_data.Date
news_data = news_data.groupby(pd.TimeGrouper('M')).agg({'combined_news': ''.join}).reset_index()


news_data['Date'] = news_data['Date'].map(lambda x: x.strftime('%Y-%m'))

news_commodity = pd.merge(commodity_prices, news_data, on=['Date'])

positive_market = news_commodity[news_commodity['Status'] == 1]
negative_market = news_commodity[news_commodity['Status'] == 0]


def to_words(content):
	letters_only = re.sub("[^a-zA-Z]", " ", content)
	words = letters_only.lower().split()
	stops = set(stopwords.words("english"))
	meaningful_words = [w for w in words if not w in stops]
	return ( " ".join(meaningful_words))

test_corpus = news_commodity['combined_news'][0].split('b')


positive_market_words = []
negative_market_words = []

for each in positive_market['combined_news']:
	positive_market_words.append(to_words(each))

for each in negative_market['combined_news']:
	negative_market_words.append(to_words(each))

positive_word_cloud = WordCloud(background_color='white',
								width=3000,
								height=2500).generate(positive_market_words[0])

plt.imshow(positive_word_cloud)
plt.axis('off')
plt.show()
stemmer = PorterStemmer()

def text_preprocessing(headlines):
	for headline in range(len(headlines)):
		tokens = nltk.word_tokenize(headlines[headline])
		try:
			singles = [stemmer.stem(token) for token in tokens]
		except:
			pass
		singles = ' '.join(singles)
		headlines[headline] = ''.join(singles)
	return headlines

#news_tokens = to_words(news_commodity['combined_news'][0]).split('b')

def tfidf_extraction(text_corpus):
	processed_text = text_preprocessing(text_corpus)
	vectorizer = TfidfVectorizer(analyzer='word')

	result = vectorizer.fit_transform(processed_text)

	feature_array = np.array(vectorizer.get_feature_names())
	tfid_sorting = np.argsort(result.toarray()).flatten()[::-1]

	n=20

	top_n = feature_array[tfid_sorting][:n]

	return top_n

def keyword_generation(row):
	return ','.join(tfidf_extraction(to_words(row['combined_news']).split(',b')))

news_commodity['Keywords'] = news_commodity.apply(keyword_generation, axis=1) 
news_commodity = news_commodity.drop('combined_news', 1)
news_commodity.to_csv("visualisation_data.csv")
