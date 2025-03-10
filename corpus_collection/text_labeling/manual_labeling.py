import pandas as pd 
import json 
import os
import readchar 

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

def save_progress(last_index, progress_file):
    """進捗をJSONファイルに保存する関数
    Args:
        last_index (int): 最後にラベル付けしたデータのインデックス
        progress_file (str): 進捗を保存するJSONファイルのパス
    """
    try:
        progress_data = { # 進捗データを格納する辞書
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

            selected_corpus = progress_data["selected_corpus"] # 選択されたコーパス
            last_index = progress_data["last_index"] # 最後にラベル付けしたデータのインデックス

            print(f"進捗が{progress_file}から読み込まれました。") 
            return selected_corpus, last_index
        except Exception as e:
            print(f"進捗の読み込み中にエラーが発生しました: {e}") 
    return 0 

def label_data(file_path, last_index, progress_file_absolutepath):
    """データにラベルを付ける関数
    Args:
        file_path (str): ラベル付けするコーパスCSVの絶対パス
        last_index (int): 最後にラベル付けしたデータのインデックス
    Returns:
        pandas.DataFrame: ラベル付けされたデータ
    """
    data = load_csv(file_path) # CSVファイルを読み込む
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
        print("ラベルを入力してください (1: 皮肉 2: ポジティブ 3: ネガティブ 4: ニュートラル 5: ノイズ q: 中断):") # ラベルの選択肢を表示
        while True: # 無限ループ
            label = readchar.readchar() # キー入力を読み取る
            if label == "q": # qが入力された場合
                flag = True # 中断フラグをTrueにする
                save_progress(index, progress_file_absolutepath) # 進捗を保存
                break # ループを抜ける
            elif str(label) in ["1", "2", "3", "4", "5"]: # 1から5の数字が入力された場合
                data.at[index, 'label'] = int(label) # labelを更新
                break # ループを抜ける
            else:
                print("1から5の数字を入力してください") # エラーメッセージを表示
    return data # ラベル付けされたデータを返す

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
    progress_file_absolutepath = os.path.join(selected_dir_path, "progress.json")
    if os.path.exists(progress_file_absolutepath): # ファイルが存在するか確認
        progress_file = progress_file_absolutepath
        selected_corpus, last_index = load_progress(progress_file) # 進捗を読み込む
        file_path = os.path.join(selected_dir_path, selected_corpus) # これはコーパスの絶対パス
    else:
        # ファイルの選択
        all_files = os.listdir(selected_dir_path)
        csv_files = [f for f in all_files if f.endswith(".csv")] # CSVファイルのみを抽出
        print(csv_files)
        selected_file = str(input("ファイルを選択してください: "))
        last_index = 0
        create_progress_file(progress_file_absolutepath, selected_file, last_index)
        file_path = os.path.join(selected_dir_path, selected_file) # これはコーパスの絶対パス
        
    # progress_file = "progress.json"
    # last_index = load_progress(progress_file) # 進捗を読み込む
    # if last_index == 0: # 進捗がない場合
    #     file_path = input("CSVファイルのパスを入力してください: ") # CSVファイルのパスを入力
    # else:
    #     file_path = "combined_file.csv" # CSVファイルのパス
    # print(f"進捗: {last_index}") # 進捗を表示

    if os.path.exists(file_path):
        data = label_data(file_path, last_index, progress_file_absolutepath) # ラベル付け
        output_file_path = os.path.join(selected_dir_path, "labeled.csv") # 出力ファイルのパス
        save_csv(data, output_file_path) # CSVファイルに保存

if __name__ == "__main__":
    main() # main関数を実行