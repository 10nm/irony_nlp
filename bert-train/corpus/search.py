import csv
import os
import re
import json

pattens = [
    '(皮肉)', '（皮肉）', 'という皮肉', 'とゆう皮肉', 'とは皮肉','皮肉にも', '皮肉なこと'
]
patten_num = [
    0,0,1,1,1,2,2
]

patten_dict = {num: [] for num in set(patten_num)}

latest = ""

count = 0
with open('hiniku.tsv' , 'r')as csvfile:
    for row in csv.reader(csvfile):
        for r in row:
            for i, patten in enumerate(pattens):
                if patten in r:
                    if latest == patten_num[i]:
                        continue
                    print(patten_num[i])
                    latest = patten_num[i]

                    patten_dict[patten_num[i]].append(r)
                    break
    print(count)

print(patten_dict)

with open('patten_dict.json', 'w', encoding='utf-8') as jsonfile:
    json.dump(patten_dict, jsonfile, ensure_ascii=False, indent=4)