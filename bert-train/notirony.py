file_path = './corpus/newsplus.tsv'
search_terms = ['(皮肉)', '（皮肉）', 'という皮肉', 'とゆう皮肉','皮肉にも', '皮肉なこと', 'とは皮肉']
import time

targets = []
replies = []
flag = False
prev = ""
count = 0
def read_file_line_by_line(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines

def search_and_remove(line):
    global count
    global flag
    if flag:
        return None
    global prev
    line = str(line).split('\t')
    r = len(line)
    if r == 1:
        return None, None
    l = line[0]

    if "皮肉" in l:
        return None
    if l == prev:
        return None
    prev = l
    
    l = str(l).replace("__BR__", "")

    return l
                
                

def get_irony_lines(file_path):
    lines = read_file_line_by_line(file_path)
    for l in lines:

        target  = search_and_remove(l)
        if (target is not None) :
            targets.append(target)
    


# def save_csv(data, (str(file_path) + str(file_name))):
#     with open(file_name, 'w') as file:
#         for line in data:
#             file.write(line + '\n')
#     print("Data saved successfully.")


get_irony_lines(file_path)
print("Targets: ", len(targets))

# save targets and replies to a file
file_name = "notironyRE.txt"
with open(file_name, 'w') as file:
    for i in range(len(targets)):
        file.write(str(targets[i]) + '\n')
print("Data saved successfully.")




# file_path = 'saves/IronyinALL'
# lines = read_file_line_by_line(file_path)
# recent = ""
# for line in lines:
#     if len(line) != 1:
#         line = line.split('\t')
#     if line is not None:
#         l = search_and_remove(list(line))
#     if l:
#         lines.append(l)
