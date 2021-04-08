import csv
import joblib
from pprint import pprint


def dump_data_to_pkl(input_filename, output_filename, num_rows=-1):
    data_dict = []
    with open(input_filename, 'r') as rfile:
        reader = csv.reader(rfile)
        for index, row in enumerate(reader):
            # skip header row
            if index == 0:
                continue

            if num_rows > 0:
                if index - 1 >= num_rows:
                    break

            if index % 50000 == 0:
                print("{} rows processed!".format(index))

            title_text = row[1]
            body_text = row[2]
            # the tags are separated by space. Multiword tags are connected by hyphen
            tags = row[3].split()
            data_dict.append({
                'title': title_text,
                'body': body_text,
                'tags': tags
            })

    print("Total number of rows in data: %d", len(data_dict))
    pprint(data_dict[-1])
    joblib.dump(data_dict, output_filename)


if __name__ == '__main__':
    filename_csv = "../data/Train.csv"
    filename_pkl = "../data/training_data.pkl"
    dump_data_to_pkl(filename_csv, filename_pkl, num_rows=50000)
