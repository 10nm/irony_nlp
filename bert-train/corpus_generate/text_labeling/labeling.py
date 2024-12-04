import pandas as pd
import json
import os
import readchar

def load_csv(file_path):
    """CSVファイルを読み込み、データフレームを返す関数"""
    try:
        data = pd.read_csv(file_path)
        print(f"{file_path} loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"file not found: {file_path}")
    except Exception as e:
        print(f"error: {e}")
    return None

def save_progress(last_index, progress_file):
    """進捗をJSONファイルに保存する関数"""
    try:
        progress_data = {
            "last_index": last_index
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)
        print(f"進捗が{progress_file}に保存されました。")
    except Exception as e:
        print(f"進捗の保存中にエラーが発生しました: {e}")

def load_progress(progress_file):
    """JSONファイルから進捗を読み込む関数"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            last_index = progress_data["last_index"]
            print(f"進捗が{progress_file}から読み込まれました。")
            return last_index
        except Exception as e:
            print(f"進捗の読み込み中にエラーが発生しました: {e}")
    return 0

def label_data(data, progress_file, start_index):
    """データにラベルを付ける関数"""
    flag = False
    if 'label' not in data.columns:
        data['label'] = None

    ## start_indexから最終行までのデータ , iterrows()で行を取得 => (index, row)でindexと行を取得
    for index, row in data.iloc[start_index:].iterrows():
        if flag: 
            break
        # notna : NaN -> False
        if pd.notna(row['label']):
            continue
        print(f"index: {index}")
        print(row["Utterance"].replace("\n", " "), end=" | "); print(row["Response"].replace("\n", " "))
        print("ラベルを入力してください (1: 皮肉 2: ポジティブ 3: ネガティブ 4: ニュートラル 5: ノイズ q: 中断):")
        while True:
            label = readchar.readchar()
            if label == "q":
                flag = True
                save_progress(index, progress_file)
                break
            elif str(label) in ["1", "2", "3", "4", "5"]:
                data.at[index, 'label'] = int(label)
                break
            else:
                print("1から5の数字を入力してください")
    return data

def save_csv(data, output_file_path):
    """データをCSVファイルに保存する関数"""
    try:
        data.to_csv(output_file_path, index=False)
        print(f"データが{output_file_path}に保存されました。")
    except Exception as e:
        print(f"データの保存中にエラーが発生しました: {e}")

def main():
    progress_file = "progress.json"
    last_index = load_progress(progress_file)
    if last_index == 0:
        file_path = input("CSVファイルのパスを入力してください: ")
    else:
        file_path = "save.csv"
    print(f"進捗: {last_index}")
    data = load_csv(file_path)
    if data is not None:
        data = label_data(data,progress_file, last_index)
        output_file_path = "save.csv"
        save_csv(data, output_file_path)

if __name__ == "__main__":
    main()