import csv
import readchar
import sys

# CSVフィールドサイズの制限を増加
csv.field_size_limit(sys.maxsize)

#### 対話データセットを読み込み、2列目(1列目の発話に対する応答)に
#### 特定の文字列が含まれている行に対してラベル付け/整形を行うスクリプト

def search_and_label(input_file, output_file, search_strings):
    # 入力CSVファイルを読み込み
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        if 'label' not in reader.fieldnames:
            reader.fieldnames.append('label')
            for row in rows:
                row['label'] = None
    
    # 1 : 2項目のみを検索対象
    # 2 : 1・2項目を検索対象
    search_target = int(input("Search target (1 or 2): "))
    flag = False
    # ラベル付けと2列目の書き換えのための処理
    HIT_COUNT = 0
    ONE_COUNT = 0
    flag = False
    for row in rows:
        if flag:
            break
        try:
            label = int(float(row['label']))
            continue
        except:
            label = None
        # 一致した検索文字列を保持する変数
        found_string = None

        # Response列での検索
        for search_string in search_strings:
            if search_string in row['Response']:
                found_string = search_string
                break

        # Utterance列での検索（search_target == 2の場合）
        if found_string is None and search_target == 2:
            for search_string in search_strings:
                if search_string in row['Utterance']:
                    found_string = search_string
                    break
        # 一致した場合の処理
        if found_string is not None:
            HIT_COUNT += 1
            resp = row['Response']
            print(f"Count: {HIT_COUNT}, {ONE_COUNT}")
            print(f"Utterance: {row['Utterance']} \n Response: {resp}")  # UtteranceとResponseの内容をユーザーに提示
            print("Enter label for this row (1-5): ", end='', flush=True)
            while True:
                label = readchar.readkey()
                if label in ('12345q'):  # 1から4のいずれかの入力を受け付ける
                    break
            
            if str(label) == 'q':
                flag = True
                break
            elif str(label) == '1':  # ラベルが1の場合のみ2列目の値を置き換える
                ONE_COUNT += 1
                row['label'] = label  
                colum_replacement = input("Enter new value for Response: ")  # 2列目の新しい値を入力
                if colum_replacement != "":  # 空白（Enterのみ）の場合は変更しない
                    row['Response'] = colum_replacement  # 2列目の値を新しい値に置き換え
                else:
                    row['Response'] = resp
            else:
                row['label'] = label 

        else:
            pass  # 一致しない場合は何もしない
    
    # 新しいCSVファイルに保存 
    with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# search_strings = ['(皮肉)', '（皮肉）', '(皮肉', '（皮肉']
# search_strings = ['(嘲)', '（嘲）', '(嘲', '（嘲']
# search_strings = ['(笑)', '（笑）', '(笑', '（笑']
search_strings = ['素晴らしい']
search_and_label('./work/1_n4v/n4v.csv', './work/1_n4v/n4v.csv', search_strings)
search_and_label('./work/2_ljp/ljp.csv', './work/2_ljp/ljp.csv', search_strings)
search_and_label('./work/3_nwp/nwp.csv', './work/3_nwp/nwp.csv', search_strings)
