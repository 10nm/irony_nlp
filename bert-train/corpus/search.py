import csv
import os
import re

pattens = [
    '(皮肉)', '（皮肉）', 'という皮肉', 'とゆう皮肉', 'とは皮肉','皮肉にも', '皮肉なこと'
]
patten_num = [
    0,0,1,1,1,2,2
]

patten_dict = {num: [] for num in set(patten_num)}

count = 0
with open('hiniku.tsv' , 'r')as csvfile:
    for row in csv.reader(csvfile):
        for r in row:
            for i, patten in enumerate(pattens):
                if patten in r:
                    print(patten_num[i])
                    patten_dict[patten_num[i]].append(r)
                    break
    print(count)

print(patten_dict)
