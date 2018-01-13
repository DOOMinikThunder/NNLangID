from random import shuffle
import unicodecsv as csv

class DataSplit(object):

	def split_percent_of_languages(self, input_file, ratio, out_filenames):
		if(len(ratio) != len(out_filenames)):
			print("ratio and output files must have same size!")
			return []
		texts_and_languages = self.read_input(input_file)
		languages_splitted = self.split_by_languages(texts_and_languages)
		splitted_data = self.merge_splitted_languages(languages_splitted, ratio)
		self.write_to_file(splitted_data,out_filenames)
		return splitted_data

	def write_to_file(self, splitted_data, out_filenames):
		for list, out_file in zip(splitted_data, out_filenames):
			with open(out_file, 'wb') as file:
				writer = csv.writer(file, delimiter=';')
				for tweet in list:
					to_write = [tweet[0], tweet[1], tweet[2]]
					#print(to_write)
					writer.writerow(to_write)

	def read_input(self, csv_file):
		with open(csv_file, 'rb') as file:
			reader = csv.reader(file, delimiter=';', encoding='utf-8')
			next(reader)
			data = [row[:] for row in reader]
		return data


	def merge_splitted_languages(self, languages_splitted, ratio):
		splitted_data = [[] for i in range(len(ratio))]
		for language in languages_splitted:
			language_in_ratios = self.split_language_with_ratio(languages_splitted[language], ratio)#
			for i,set in enumerate(language_in_ratios):
				splitted_data[i] += set

		return splitted_data

	def split_by_languages(self, input):
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
	def split_language_with_ratio(self, input, ratio):
		in_size = len(input)
		idx_list = [i for i in range(len(input))]
		shuffle(idx_list)
		split_languages = []
		for percentage in ratio[:-1]:
			split_languages.append(self.take_from_list(input, 0, int(in_size*percentage)))
		split_languages.append(input)
		return split_languages

	def take_from_list(self, list, start, end):
		new_list = list[start:end]
		del list[start:end]
		return new_list