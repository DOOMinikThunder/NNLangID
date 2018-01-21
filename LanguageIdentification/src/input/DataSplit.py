# -*- coding: utf-8 -*-

#    MIT License
#    
#    Copyright (c) 2018 Alexander Heilig, Dominik Sauter, Tabea Kiupel
#    
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#    
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#    
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.


from random import shuffle, seed
import unicodecsv as csv


class DataSplit(object):
    """Class for splitting an original file into separate training, validation and test set files.
    """
    
    def split_percent_of_languages(self, input_file, ratio, out_filenames, shuffle_seed):
        """
        Splits all tweets from input files into different sets,
        each set contains a percent (ratio) of the original file's tweets.
        
        Args:
            input_file: contains all data to be splitted
            ratio: list of ratios, determines the number of output files
            out_filenames: list of output file names
            shuffle_seed: ensures the same splitup if files are created more than once

        Returns:
            splitted_data: list splitted data sets
        """
        if(len(ratio) != len(out_filenames)):
            print("ratio and output files must have same size!")
            return []
        texts_and_languages = self.__read_input(input_file)
        languages_splitted = self.__split_by_languages(texts_and_languages)
        splitted_data = self.__merge_splitted_languages(languages_splitted, ratio, shuffle_seed)
        self.__write_to_file(splitted_data,out_filenames)
        return splitted_data

    def __write_to_file(self, splitted_data, out_filenames):
        """
        writes the splitted data into new files
        overrides existing files
        
        Args:
            splitted_data: list of splitted sets
            out_filenames: output filenames
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

        Returns:
            data: data as list
        """
        with open(csv_file, 'rb') as file:
            reader = csv.reader(file, delimiter=';', encoding='utf-8')
            next(reader)
            data = [row[:] for row in reader]
        return data

    def __merge_splitted_languages(self, languages_splitted, ratio, shuffle_seed):
        """
        Splits languages in ratios and merges the respective ratios.
       
        Args:
            languages_splitted: list for each language
            ratio: determines the ratios each language will be splitted in
            shuffle_seed: shuffles the languages itself

        Returns:
            splitted_data: list of lists containing all tweets splitted up in ratios keeping the original language distribution
        """
        splitted_data = [[] for i in range(len(ratio))]
        for language in languages_splitted:
            language_in_ratios = self.split_data_with_ratio(languages_splitted[language], ratio, shuffle_seed)#
            for i,set in enumerate(language_in_ratios):
                splitted_data[i] += set

        return splitted_data

    def __split_by_languages(self, input):
        """
        splits input list in dict with entries 'language': tweets with this language
        
        Args:
            input: input tweets and language

        Returns:
             languages_splitted: dict containing the splitted languages
        """
        languages_splitted = {}
        for tweet in input:
            if tweet[2] in languages_splitted:
                languages_splitted[tweet[2]].append(tweet)
            else:
                languages_splitted[tweet[2]] = [tweet]
        return languages_splitted

    def split_data_with_ratio(self, input, ratio, shuffle_seed):
        """
        Splits a list of dates into ratio parts.
        
        Args:
            input: data to split into ratio
            ratio: list of ratios the data will be splitted into,
                e.g. [0.5, 0.5] for two lists, [0.4, 0.4, 0.2] for three lists
            shuffle_seed: seed for shuffling the data

        Returns:
            split_data: list of data splitted into len(ratio) parts
        """
        in_size = len(input)
        seed(shuffle_seed)
        shuffle(input)
        split_data = []
        indices = [0]
        for i,percentage in enumerate(ratio[:-1]):
            indices.append(int(in_size*percentage)+indices[i])
        indices.append(in_size)
        for i,j in zip(indices[:-1], indices[1:]):
            split_data.append(input[i:j])
        return split_data
