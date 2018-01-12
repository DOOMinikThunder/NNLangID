from random import shuffle
from . import InputData

class DataSplit(object):
	def __init__(self):
		input_data_reader = InputData.InputData()
		pass

	def split_percent_of_languages(self, input_file, ratio):
		pass

	def split_by_languages(self, input):
		idx = 0
		end_of_list = len(input)-1
		language_splitted = []
		while(idx != end_of_list):
			take_language = input[idx][1]
			temp_list = []
			while(take_language == input[idx][1]):
				temp_list.append(input[idx])
				idx += 1
			language_splitted.append(temp_list)

		pass

	#ratio is a list of ratios, e.g. [0.5,0.5] for two lists, [0.4,0.4,0.2] for three lists
	#returns list of split lists with len(ratio)
	def split_language_with_ratio(self, input, ratio):
		in_size = len(input)
		idx_list = [i for i in range(len(input))]
		shuffle(idx_list)
		split_languages = []
		for percentage in ratio[:-1]:
			split_languages.append(self.take_from_list(input, 0, in_size*percentage))
		split_languages.append(input)
		return split_languages

	def take_from_list(self, list, start, end):
		new_list = list[start:end]
		del list[start, end]
		return new_list