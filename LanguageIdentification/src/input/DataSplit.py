from random import shuffle, seed
import unicodecsv as csv

class DataSplit(object):

	def split_percent_of_languages(self, input_file, ratio, out_filenames, shuffle_seed):
		"""
		Splits all tweets from input files into different sets
		each set contains a percent (ratio) of the original file's tweets
		Args:
			input_file: contains all data to be splitted
			ratio: list of ratios, determines the number of output files
			out_filenames: list of output file names
			shuffle_seed: ensures the same splitup if files are created more than once

		Returns: list splitted data sets

		"""
		if(len(ratio) != len(out_filenames)):
			print("ratio and output files must have same size!")
			return []
		texts_and_languages = self.read_input(input_file)
		languages_splitted = self.split_by_languages(texts_and_languages)
		splitted_data = self.merge_splitted_languages(languages_splitted, ratio, shuffle_seed)
		self.write_to_file(splitted_data,out_filenames)
		return splitted_data

	def __write_to_file(self, splitted_data, out_filenames):
		"""
		writes the splitted data into new files
		overrides existing files
		Args:
			splitted_data: list of splitted sets
			out_filenames: output filenames

		Returns:

		"""
		for list, out_file in zip(splitted_data, out_filenames):
			with open(out_file, 'wb') as file:
				writer = csv.writer(file, delimiter=';')
				for tweet in list:
					to_write = [tweet[0], tweet[1], tweet[2]]
					#print(to_write)
					writer.writerow(to_write)

	def __read_input(self, csv_file):
		"""
		reads the data from a csv file

		Args:
			csv_file: input csv file

		Returns: data as list

		"""
		with open(csv_file, 'rb') as file:
			reader = csv.reader(file, delimiter=';', encoding='utf-8')
			next(reader)
			data = [row[:] for row in reader]
		return data


	def __merge_splitted_languages(self, languages_splitted, ratio, shuffle_seed):
		"""
		Splits languages in ratios and merge the respective ratios
		Args:
			languages_splitted: list for each language
			ratio: determines the ratios each language will be splitted in
			shuffle_seed: shuffles the languages itself

		Returns:

		"""
		splitted_data = [[] for i in range(len(ratio))]
		for language in languages_splitted:
			language_in_ratios = self.split_language_with_ratio(languages_splitted[language], ratio, shuffle_seed)#
			for i,set in enumerate(language_in_ratios):
				splitted_data[i] += set

		return splitted_data

	def __split_by_languages(self, input):
		"""
		splits input list in dict with entries 'language': tweets with this language
		Args:
			input: input tweets and language

		Returns: dict containing the splitted languages

		"""
		idx = 0
		end_of_list = len(input)-1
		languages_splitted = {}
		for tweet in input:
			if tweet[2] in languages_splitted:
				languages_splitted[tweet[2]].append(tweet)
			else:
				languages_splitted[tweet[2]] = [tweet]
		return languages_splitted

	#ratio is a list of ratios, e.g. [0.5,0.5] for two lists, [0.4,0.4,0.2] for three lists
	#returns list of split lists with len(ratio)
	def __split_language_with_ratio(self, input, ratio, shuffle_seed):
		"""
		Splits a list of tweets of one language into ratio parts
		Args:
			input: list of tweets one language
			ratio: ratio the tweets will be splitted into
			shuffle_seed: shuffles the tweets

		Returns: list of len(ratio) language splitted into ratio parts

		"""
		in_size = len(input)
		idx_list = [i for i in range(len(input))]
		seed(shuffle_seed)
		shuffle(idx_list)
		split_languages = []
		for percentage in ratio[:-1]:
			split_languages.append(self.take_from_list(input, 0, int(in_size*percentage)))
		split_languages.append(input)
		return split_languages

	def __take_from_list(self, list, start, end):
		"""
		retrieves elements from list and deletes the retrieved elements
		Args:
			list: input list
			start: start index
			end: end index, not included

		Returns: input list with list[start:end] deleted

		"""
		new_list = list[start:end]
		del list[start:end]
		return new_list