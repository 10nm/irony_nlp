import csv
import sys
import time 
import os
import re
count = 0
data = []

def save_csv(data, file_name):
        file_exists = os.path.isfile(file_name)
        directory = os.path.dirname(file_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        if not file_exists:
            open(file_name, 'w').close()

        with open(file_name, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(data)


def search_csv(file_paths, search_terms):
    csv.field_size_limit(sys.maxsize)
    for f in file_paths:
        with open(f, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    for s in search_terms:
                            if str(s) in str(row[0]):
                                print(row)
                                data.append(row)

# file_paths = ['corpus/livejupiter.tsv', 'corpus/news4vip.tsv', 'corpus/newsplus.tsv']
# search_terms = ('(皮肉)', '（皮肉）', 'という皮肉', 'とゆう皮肉','皮肉にも', '皮肉なこと', 'とは皮肉') 

search_csv(file_paths, search_terms)

print(count)
file_name = str(input("Enter the file name to save the data: "))
save_csv(data, ("saves/" + str(file_name)))
