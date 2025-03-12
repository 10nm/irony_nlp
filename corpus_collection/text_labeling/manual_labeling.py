import pandas as pd 
import json 
import os
import readchar 
import os, csv, re, time, json, signal, sys, pandas as pd

api_key = os.getenv("GEMINI_API_KEY")

import google.generativeai as genai
genai.configure(api_key=api_key)
 
model = genai.GenerativeModel("gemini-2.0-flash-lite")

def load_csv(file_path):
    """CSVファイルを読み込み、データフレームを返す関数
    Args:
        file_path (str): CSVファイルのパス
    Returns:
        pandas.DataFrame: CSVファイルの内容を格納したデータフレーム。
                          ファイルが存在しない場合や読み込みに失敗した場合はNoneを返す。
    """
    try:
        data = pd.read_csv(file_path) # CSVファイルを読み込む
        print(f"{file_path} loaded successfully.") 
        return data # データフレームを返す
    except FileNotFoundError:
        print(f"file not found: {file_path}") # ファイルが存在しない場合
    except Exception as e:
        print(f"error: {e}") # その他のエラー
    return None # エラーが発生した場合、Noneを返す

def save_progress(last_index, progress_file, selected_corpus):
    """進捗をJSONファイルに保存する関数
    Args:
        last_index (int): 最後にラベル付けしたデータのインデックス
        progress_file (str): 進捗を保存するJSONファイルのパス
        selected_corpus (str): 選択されたコーパスの名前
    """
    try:
        progress_data = { # 進捗データを格納する辞書
            "selected_corpus": selected_corpus, # 選択されたコーパスの名前
            "last_index": last_index # 最後にラベル付けしたデータのインデックス
        }
        with open(progress_file, 'w') as f: # JSONファイルを開く
            json.dump(progress_data, f) # JSONファイルに書き込む
        print(f"進捗が{progress_file}に保存されました。") # 保存成功メッセージ
    except Exception as e:
        print(f"進捗の保存中にエラーが発生しました: {e}") # エラーが発生した場合

def load_progress(progress_file):
    """JSONファイルから進捗を読み込む関数
    Args:
        progress_file (str): 進捗を保存したJSONファイルのパス
    Returns:
        tuple: (選択されたコーパスの名前, 最後にラベル付けしたデータのインデックス)。
               JSONファイルが存在しない場合や読み込みに失敗した場合は0を返す。
    """
    if os.path.exists(progress_file): 
        try:
            with open(progress_file, 'r') as f: 
                progress_data = json.load(f) 
            print(f"進捗が{progress_file}から読み込まれました。")
            selected_corpus = progress_data["selected_corpus"] # 選択されたコーパス
            last_index = progress_data["last_index"] # 最後にラベル付けしたデータのインデックス

            print(f"進捗が{progress_file}から読み込まれました。") 
            return selected_corpus, last_index
        except Exception as e:
            print(f"進捗の読み込み中にエラーが発生しました: {e}")
    return None, 0 # ファイルが存在しない場合はNoneと0を返す

def allow_or_deny():
    """許可または拒否を選択する関数
    Returns:
        bool: 許可の場合はTrue、拒否の場合はFalse
    """
    print("承認しますか。 (a: 承認, d: 拒否):") # 選択肢を表示
    while True:
        key = readchar.readchar() # キー入力を読み取る
        if key == "a": 
            return True
        elif key == "d": 
            return False
        else:
            print("aまたはdを入力してください")

def labeling_manual(data, index):
    """手動ラベリングを行う関数
    Args:
        data (pandas): ラベリングするデータ
        index (int): ラベル付けするデータのインデックス
    Returns:
        is_break (bool): 中断フラグ
    """
    print("ラベルを入力してください (1: 皮肉 2: ポジティブ 3: ネガティブ 4: ニュートラル 5: ノイズ q: 中断):") # ラベルの選択肢を表示
    while True: # 無限ループ
        label = readchar.readchar() # キー入力を読み取る
        if label == "q": # qが入力された場合
            is_break = True
            return is_break
        elif str(label) in ["1", "2", "3", "4", "5"]: # 1から5の数字が入力された場合
            data.at[index, 'label'] = int(label) # labelを更新
            is_break = False
            return is_break
        else:
            print("1から5の数字を入力してください") # エラーメッセージを表示

def labeling_auto(data):
    """ 自動ラベリングを行う関数
    Args:
        data (str): ラベリングするデータ
    
    Returns:
        str: ラベリング結果 (0: ironic, 1: not ironic, 2: error)
    """
    utr = data["Utterance"].replace("\n", " ")
    res = data["Response"].replace("\n", " ")

    prompt = f"""
# 皮肉判定タスク
以下の対話から、発話者Aに対するBの返答が皮肉を含むか否かを判定し、<ironic> </ironic>のタグで結果を囲んだ、
<ironic>yes</ironic> または <ironic>no</ironic> で応答してください。
固有名詞などは超簡潔に説明も入れてください。

# 対話
A: {utr}
B: {res}
"""
    response = model.generate_content(prompt)
    response = response.text
    print(response)
    pattern = r"<ironic>(.*?)</ironic>"
    matches = re.findall(pattern, response, re.DOTALL)
    print(matches)
    if matches:
        if matches[0] == "yes":
            return "0"
        elif matches[0] == "no":
            return "1"
        else:
            return "2"
    else:
        return "2"

def label_data(file_path, last_index):
    """データにラベルを付ける関数
    Args:
        file_path (str): ラベル付けするコーパスCSVの絶対パス
        last_index (int): 最後にラベル付けしたデータのインデックス
    Returns:
        pandas.DataFrame: ラベル付けされたデータ
        index (int): 最後にラベル付けしたデータのインデックス
    """
    data = load_csv(file_path) # CSVファイルを読み込む
    if data is None:
        return None, last_index
    flag = False # 中断フラグ
    if 'label' not in data.columns: # label列が存在しない場合
        data['label'] = None # label列を追加

    ## start_indexから最終行までのデータ , iterrows()で行を取得 => (index, row)でindexと行を取得
    for index, row in data.iloc[last_index:].iterrows(): # データフレームの行を順番に処理
        if flag: # 中断フラグがTrueの場合
            break # ループを抜ける
        # notna : NaN -> False
        if pd.notna(row['label']): # labelがNaNでない場合
            continue # 次の行へ
        print(f"index: {index}") # インデックスを表示
        print(row["Utterance"].replace("\n", " "), end=" | "); print(row["Response"].replace("\n", " ")) # UtteranceとResponseを表示
        
        # 自動でのラベリング
        label = labeling_auto(row)
        if label == "2": # エラーが発生した場合
            is_break = labeling_manual(data, index) # 手動ラベリング
            if is_break:
                flag = True
        elif label == "0" or label == "1":
            if allow_or_deny():
                data.at[index, 'label'] = int(4) # labelを更新
            else:
                is_break = labeling_manual(data, index)
                if is_break:
                    flag = True
        else:
            print("不正な状態です")
            is_break = labeling_manual(data, index)
            if is_break:
                flag = True

    return data, index # ラベル付けされたデータを返す

def save_csv(data, output_file_path):
    """データをCSVファイルに保存する関数
    Args:
        data (pandas.DataFrame): 保存するデータ
        output_file_path (str): CSVファイルのパス
    """
    try:
        data.to_csv(output_file_path, index=False) # CSVファイルに保存
        print(f"データが{output_file_path}に保存されました。") # 保存成功メッセージ
    except Exception as e:
        print(f"データの保存中にエラーが発生しました: {e}") # エラーが発生した場合

def create_progress_file(progress_file_absolutepath, selected_file, last_index=0):
    """進捗ファイルを作成する関数
    Args:
        progress_file_absolutepath (str): 進捗ファイルの絶対パス
        selected_file (str): 選択されたファイル名
        last_index (int): 最後にラベル付けしたデータのインデックス
    """
    progress_data = {   
        "selected_corpus": selected_file,
        "last_index": last_index
    }
    with open(progress_file_absolutepath, 'w') as f:
        json.dump(progress_data, f)

def main():

    Absolute_path = os.path.dirname(__file__) # 実行ファイルのディレクトリパス
    Work_path = os.path.join(Absolute_path, "work") # 作業ディレクトリのパス

    # ディレクトリ内の候補(数字からはじまるディレクトリをリストアップ)
    all_items = os.listdir(Work_path) # work/ディレクトリ内のファイルとディレクトリを取得
    directories = [d for d in all_items if os.path.isdir(os.path.join(Work_path, d))] # ディレクトリのみを抽出
    numeric_dirs = [d for d in directories if d[0].isdigit()] # 数字から始まるディレクトリのみを抽出

    # ディレクトリの選択
    print(numeric_dirs) # ディレクトリのリストを表示
    selected_dir = str(input("ディレクトリを選択してください: ")) # ディレクトリを選択
    selected_dir_path = os.path.join(Work_path, selected_dir) # 選択されたディレクトリのパス

    # 進捗を確認
    ## progress.jsonの存在を確認
    global progress_file_absolutepath
    progress_file_absolutepath = os.path.join(selected_dir_path, "progress.json")
    print(progress_file_absolutepath)
    selected_corpus, last_index = load_progress(progress_file_absolutepath) # 進捗を読み込む
    
    # selected_corpusがNoneの場合の処理を追加
    if selected_corpus is None:
        # ファイルの選択
        all_files = os.listdir(selected_dir_path)
        csv_files = [f for f in all_files if f.endswith(".csv")] # CSVファイルのみを抽出
        print(csv_files)
        selected_file = str(input("ファイルを選択してください: "))
        last_index = 0
        create_progress_file(progress_file_absolutepath, selected_file, last_index)
    else:
        selected_file = selected_corpus
    
    file_path = os.path.join(selected_dir_path, selected_file) # これはコーパスの絶対パス
    print(f"選択されたファイル: {file_path}")

    if os.path.exists(file_path):
        data, index = label_data(file_path, last_index) # ラベル付け
        if data is not None:
            output_file_path = os.path.join(selected_dir_path, "labeled.csv") # 出力ファイルのパス
            save_csv(data, output_file_path) # CSVファイルに保存
            save_progress(int(index)-1, progress_file_absolutepath, selected_file)
    else:
        print(f"ファイルが存在しません: {file_path}")

if __name__ == "__main__":
    main()