import os
import re
import csv
import joblib
from random import random
from pprint import pprint
from collections import defaultdict
import matplotlib.pyplot as plt

import config


def dump_data_to_pkl(input_filename, output_filename, num_rows=-1, randomize=False):

    def parse_row(row):
        title_text = row[1]
        body_text = row[2]
        # the tags are separated by space. Multiword tags are connected by hyphen
        keywords = row[3].split()
        return {
            'title': title_text,
            'body': body_text,
            'keywords': keywords
        }

    data_dict = []
    with open(input_filename, 'r') as rfile:
        reader = csv.reader(rfile)
        num_rows_processed = 0
        for index, row in enumerate(reader):
            # skip header row
            if index == 0:
                continue

            if num_rows > 0:
                if num_rows_processed >= num_rows:
                    break

            if index % 50000 == 0:
                print("{} rows processed!".format(index))

            if randomize:
                selection_prob = random()
                if selection_prob >= 0.5:
                    data_dict.append(parse_row(row))
                    num_rows_processed += 1
            else:
                data_dict.append(parse_row(row))
                num_rows_processed += 1

    print("Total number of rows in data: ", len(data_dict))
    pprint(data_dict[-1])
    joblib.dump(data_dict, output_filename)


def identify_unique_tags(data):
    tags = set()
    tags_rg = '<\\w+>'
    for i in range(len(data)):
        body_text = data[i]['body']
        tags_in_text = re.findall(tags_rg, body_text)
        tags_in_text = [a.lower() for a in tags_in_text]
        tags = tags.union(tags_in_text)

    print(', '.join(tags))


def get_keyword_counts(data):
    kw_counts = defaultdict(lambda: 0)
    for elem in data:
        for kw in elem['keywords']:
            kw_counts[kw] += 1

    pprint(kw_counts)
    print("Number of keywords in dataset: {}".format(len(kw_counts.keys())))
    print("Size of data: {}".format(len(data)))

    return kw_counts


def filter_data_by_counts(data, kw_counts, count_threshold=50):
    # here we remove those keywords whose count is less than count_threshold
    for i in range(len(data)):
        data[i]['keywords'] = [kw for kw in data[i]['keywords'] if kw_counts[kw] >= count_threshold]

    # here we remove those elements in the data which don't have any keywords left
    data[:] = [elem for elem in data if len(elem['keywords']) > 0]
    print("{} records remain after filtering on threshold {}".format(len(data), count_threshold))
    return data


if __name__ == '__main__':
    if not os.path.exists(config.FILENAME_PKL):
        dump_data_to_pkl(config.FILENAME_CSV, config.FILENAME_PKL, num_rows=config.NUM_ROWS_TO_LOAD, randomize=True)

    data = joblib.load(config.FILENAME_PKL)
    pprint(data[-1])
    # identify_unique_tags(data)
    kw_counts = get_keyword_counts(data)

    data = filter_data_by_counts(data, kw_counts)
    kw_counts = get_keyword_counts(data)
    joblib.dump(data, config.FILENAME_PKL)
    plt.hist(list(kw_counts.values()), bins=10)
    plt.show()
