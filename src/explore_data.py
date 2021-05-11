import os
import re
import csv
import joblib
from random import random
from pprint import pprint


def dump_data_to_pkl(input_filename, output_filename, num_rows=-1, randomize=False):

    def parse_row(row):
        title_text = row[1]
        body_text = row[2]
        # the tags are separated by space. Multiword tags are connected by hyphen
        tags = row[3].split()
        return {
            'title': title_text,
            'body': body_text,
            'tags': tags
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


def explore_data(filename_pkl):
    data = joblib.load(filename_pkl)


if __name__ == '__main__':
    filename_csv = "../data/Train.csv"
    filename_pkl = "../data/training_data.pkl"
    if not os.path.exists(filename_pkl):
        dump_data_to_pkl(filename_csv, filename_pkl, num_rows=100000, randomize=True)

    data = joblib.load(filename_pkl)
    identify_unique_tags(data)
