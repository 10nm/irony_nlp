import csv
import readchar

def search_and_label(input_file, output_file, search_strings):
    # 入力CSVファイルを読み込み
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
    # ラベル付けと2列目の書き換えのための処理
    for row in rows:

        try:
            label = int(float(row['label']))
        except ValueError:
            label = None
        if label in [1, 2, 3, 4]:  # 既にラベルが付いている場合はスキップ
            continue
        if any(search_string in row['Response'] for search_string in search_strings):  # 2列目に検索文字列リストのいずれかが含まれているか確認
            print(f"Utterance: {row['Utterance']} \n Response: {row['Response']}")  # UtteranceとResponseの内容をユーザーに提示
            print("Enter label for this row (1-4): ", end='', flush=True)
            while True:
                label = readchar.readkey()
                if label in '1234':  # 1から4のいずれかの入力を受け付ける
                    break
            row['label'] = label  # 手動でラベルを入力
            
            colum_replacement = input("Enter new value for Response: ")  # 2列目の新しい値を入力
            if colum_replacement != "":  # 空白（Enterのみ）の場合は変更しない
                row['Response'] = colum_replacement  # 2列目の値を新しい値に置き換え
        else:
            pass  # 一致しない場合は何もしない
    
    # 新しいCSVファイルに保存 
    with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

search_strings = ['(皮肉)','（皮肉）']
search_and_label('./save.csv', 'output.csv', search_strings)

