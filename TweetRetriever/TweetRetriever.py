#!bin/bash
try:
	import json
except:
	import simplejson as json

from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
#import csv
import unicodecsv as csv
import os

ACCESS_TOKEN = '935953290650640385-JzronD80EUdqZPtWzkMu9X36hAAkN5Z'
ACCESS_SECRET = 'lCVz26Fy3kiQY6r2EKTi82O0oJUnwOLqYxt146ihDTOD7'
CONSUMER_KEY = 'HoStbDyDPk818EkjmZnOxjZQF'
CONSUMER_SECRET = 'vFIiTUwR09hyeegT7utpGBFp0nECWFFNq24hyiViAbXTfBLXjI'

class TweetRetriever(object):
	def __init__(self):
		self.oAuth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
		#Init connection to twitter streaming api
		self.twitter_stream = TwitterStream(auth=self.oAuth)
		self.twitter = Twitter(auth=self.oAuth)

		#sample_id = 484049529764052992
		#tweet = self.twitter.statuses.show(id=sample_id)
		#tweet = self.twitter.statuses.lookup(_id=sample_id)
		#print(json.dumps(tweet))
		self.tweets = []


	def retrieve_single_tweet(self, id):
		print("looking up: %s"%id)
		tweet = self.twitter.statuses.show(id=id)
		return tweet

	def retrieve_list_of_tweets(self, ids):
		tweets = self.twitter.statuses.lookup(_id=ids)
		tweet_texts = [tweet["text"] for tweet in tweets]
		tweet_ids = [tweet["id"] for tweet in tweets]
		short_tweets = {}
		for tweet in tweets:
			short_tweets[str(tweet["id"])] = tweet["text"]
		return short_tweets

	def retrieve_sample_tweets(self, amount):
		iterator = self.twitter_stream.statuses.sample()
		return self.retrieve_from_iterator(iterator, amount)

	def retrieve_specified_track_and_language(self, amount, track, languages=None):
		iterator = self.twitter_stream.statuses.filter(track=track, language=languages)
		return self.retrieve_from_iterator(iterator, amount)

	def retrieve_from_iterator(self, iterator, amount):
		short_tweets = {}
		i = 0
		for tweet in iterator:
			if 'text' in tweet:
				short_tweets[str(tweet['id'])] = tweet['text']
				i += 1
			if i == amount:
				break
		return short_tweets



	def read_language_id_csv(self, csv_file):
		tweet_id_to_language = {}
		with open(csv_file, 'rb') as file:
			reader =  csv.reader(file, delimiter=';', encoding='utf-8')
			for row in reader:
				tweet_id_to_language[str(row[1])] = row[0]
		return tweet_id_to_language

	def read_tweets_from_file(self, csv_file):
		tweet_id_to_language = self.read_language_id_csv(csv_file)
		tweet_ids = list(tweet_id_to_language.keys())
		tweet_lang = list(tweet_id_to_language.values())
		short_tweets = {}
		tweets = []
		for i in range(0,len(tweet_ids),80):
			short_tweets.update(self.retrieve_list_of_tweets(','.join(tweet_ids[i:i+80])))

			if(i%500==0):
				print("retrieved %d tweets"%len(short_tweets))
		for id, text in short_tweets.items():
			lang = tweet_id_to_language.get(id, None)
			if(lang is not None):
				tweets.append({'id':id,'text':text,'language':lang})
			else:
				print("language is none")
		return tweets



	def filter_by_language(self, tweets, lang):
		lang_tweets = []
		for tweet in tweets:
			print(tweet['language'])
			if(tweet['language'] == lang):
				lang_tweets.append(tweet)
		return lang_tweets

	def write_to_csv(self, file, tweets):
		with open(file,'ab') as file:
			writer = csv.writer(file, delimiter=';')
			for tweet in tweets:
				to_write = [tweet['id'],tweet['text'],tweet['language']]
				print(to_write)
				writer.writerow(to_write)

	def read_downloaded_tweets(self, csv_file):
		with open(csv_file, 'rb') as file:
			data = [row for row in csv.reader(file.read().splitlines())]
		print(data[0])
		#todo
		#self.tweets.append({'id':row[0],'text':row[1],'language':row[2]})


def main():
	cur_dir = os.path.dirname(os.path.abspath(__file__))
	tweet_file = cur_dir+"/data/uniformly_sampled4.csv"
	output_file = cur_dir + "/data/uniformly_sampled_dl.csv"
	tr = TweetRetriever()

	tweets = tr.read_tweets_from_file(tweet_file)
	print(len(tweets))
	tr.write_to_csv(output_file, tweets)

	#tr.read_downloaded_tweets(output_file)

	tweet_new = tr.retrieve_single_tweet("935995896637964290")
	print(json.dumps(tweet_new))


if __name__ == '__main__':
	main()
	