import google.generativeai as genai
import pandas as pd
import json
import os
import readchar
import os
import csv
import re
import time
import json
import signal
import sys
import pandas as pd
import cohere

# Gemini APIの設定
api_key_gemini = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key_gemini)
gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")

# Cohere APIの設定
api_key = os.getenv("CO_API_KEY")

co = cohere.Client(
    api_key=api_key,
)

# Groqの設定
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def load_csv(file_path):
    """CSVファイルを読み込み、データフレームを返す関数
    Args:
        file_path (str): CSVファイルのパス
    Returns:
        pandas.DataFrame: CSVファイルの内容を格納したデータフレーム。
                          ファイルが存在しない場合や読み込みに失敗した場合はNoneを返す。
    """
    try:
        data = pd.read_csv(file_path)  # CSVファイルを読み込む
        print(f"{file_path} loaded successfully.")
        return data  # データフレームを返す
    except FileNotFoundError:
        print(f"file not found: {file_path}")  # ファイルが存在しない場合
    except Exception as e:
        print(f"error: {e}")  # その他のエラー
    return None  # エラーが発生した場合、Noneを返す


def save_progress(last_index, progress_file, selected_corpus):
    """進捗をJSONファイルに保存する関数
    Args:
        last_index (int): 最後にラベル付けしたデータのインデックス
        progress_file (str): 進捗を保存するJSONファイルのパス
        selected_corpus (str): 選択されたコーパスの名前
    """
    try:
        progress_data = {  # 進捗データを格納する辞書
            "selected_corpus": selected_corpus,  # 選択されたコーパスの名前
            "last_index": last_index  # 最後にラベル付けしたデータのインデックス
        }
        with open(progress_file, 'w') as f:  # JSONファイルを開く
            json.dump(progress_data, f)  # JSONファイルに書き込む
        print(f"進捗が{progress_file}に保存されました。")  # 保存成功メッセージ
    except Exception as e:
        print(f"進捗の保存中にエラーが発生しました: {e}")  # エラーが発生した場合


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
            selected_corpus = progress_data["selected_corpus"]  # 選択されたコーパス
            last_index = progress_data["last_index"]  # 最後にラベル付けしたデータのインデックス
            return selected_corpus, last_index
        except Exception as e:
            print(f"進捗の読み込み中にエラーが発生しました: {e}")
    return None, 0  # ファイルが存在しない場合はNoneと0を返す


def allow_or_deny():
    """許可または拒否を選択する関数
    Returns:
        bool: 許可の場合はTrue、拒否の場合はFalse
    """
    print("承認しますか。 (a: 承認, d: 拒否, q: 中断):")  # 選択肢を表示
    while True:
        key = readchar.readchar()  # キー入力を読み取る
        if key == "a":
            return True
        elif key == "d":
            return False
        elif key == "q":
            return
        else:
            print("aまたはdまたはqを入力してください")


def labeling_manual(data, index):
    """手動ラベリングを行う関数
    Args:
        data (pandas): ラベリングするデータ
        index (int): ラベル付けするデータのインデックス
    Returns:
        is_break (bool): 中断フラグ
    """
    print("ラベルを入力してください (1: 皮肉 4: 非皮肉 / ニュートラル q: 中断):")  # ラベルの選択肢を表示
    while True:  # 無限ループ
        label = readchar.readchar()  # キー入力を読み取る
        if label == "q":  # qが入力された場合
            is_break = True
            return is_break
        elif str(label) in ["1", "4"]:  # 1か4の数字が入力された場合
            data.at[index, 'label'] = int(label)  # labelを更新
            is_break = False
            return is_break
        else:
            print("1か4の数字を入力してください")  # エラーメッセージを表示

def get_chat_completion(prompt, model):
    """チャットの生成を行う関数
    Args:
        prompt (str): チャットのプロンプト
        model (str): モデル名
    Returns:
        str: チャットの生成結果
    """

    # F
    if model == "GEMINI":
        response = gemini_model.generate_content(prompt)
        response = response.text
        return response
    
    # P
    elif model == "COHERE":
        chat = co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
        )
        response = chat.chat_history[-1].message
        return response
    
    # F
    elif model == "LLAMA":
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        response = chat_completion.choices[0].message.content
        return response
    
    # F
    elif model == "DEEPSEEK":
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="deepseek-r1-distill-qwen-32b",
            stream=False,
        )
        response = chat_completion.choices[0].message.content
        return response
    
    else:
        print("error")
        return None

def extract_ironic(response):
    """アイロニックか否かを抽出する関数
    Args:
        response (str): チャットの生成結果
    Returns:
        str: アイロニックか否か (0: 皮肉, 1: 非皮肉, 9: エラー)
    """
    pattern = r"<ironic>(.*?)</ironic>"
    matches = re.findall(pattern, response, re.DOTALL)
    print(matches)
    if matches:
        if matches[0] == "yes":
            return "0"
        elif matches[0] == "no":
            return "1"
        else:
            return "9"
    else:
        return "9"

def labeling_auto(data):
    """ ラベリング候補を自動で生成する関数
    Args:
        data (str): ラベリングするデータ
    Returns:
        str: ラベル (1: 皮肉 4: 非皮肉 / ニュートラル)
    """
    utr = data["Utterance"].replace("\n", " ")
    res = data["Response"].replace("\n", " ")

    prompt = f"""
# 皮肉判定タスク
以下の対話から、発話者Aに対するBの返答が、Aの発言に対しての皮肉(アイロニー)か否かを判定してください。
結果は、<ironic> </ironic>のタグで囲んだ、<ironic>yes</ironic> または <ironic>no</ironic> のみで回答してください。
ただし、固有名詞(人名など)を含む場合はそれについての簡潔な説明を入れてください。

# 対話
A: {utr}
B: {res}
"""
    
    return_counter = 0

    print("--------------------")
    # DEEPSEEK
    deepseek_response_text = get_chat_completion(prompt, "DEEPSEEK")
    deepseek_response = extract_ironic(deepseek_response_text)
    if deepseek_response:
        return_counter += 1
    print(f"DEEPSEEK_TEXT: {deepseek_response_text}")
    print(f"DEEPSEEK_RESP: {deepseek_response}")

    # GEMINI
    gemini_response_text = get_chat_completion(prompt, "GEMINI")
    gemini_response = extract_ironic(gemini_response_text)
    print(f"GEMINI_TEXT: {gemini_response_text}")
    print(f"GEMINI_RESP: {gemini_response}")
    if gemini_response:
        return_counter += 1

    # LLAMA
    llama_response_text = get_chat_completion(prompt, "LLAMA")
    llama_response = extract_ironic(llama_response_text)
    print(f"LLAMA_TEXT: {llama_response_text}")
    print(f"LLAMA_RESP: {llama_response}")
    if llama_response:
        return_counter += 1

    if return_counter == 3:
        print(f"G: {"Y" if gemini_response == "0" else "N"} L: {"Y" if llama_response == "0" else "N"} D: {"Y" if deepseek_response == "0" else "N"}")
        print("--------------------")

        # いずれかのエラー発生時
        if gemini_response == "9" or llama_response == "9" or deepseek_response == "9":
            print("LLM_JUDGE: エラー")
            return "0"
        
        # 多数決 0,1: 皮肉, 2,3: 非皮肉
        LLM_JUDGE = int(gemini_response) + int(llama_response) + int(deepseek_response)
        if LLM_JUDGE <= 1:
            # 皮肉
            print("LLM_JUDGE: 皮肉")
            return "1"
        else:
            # 非皮肉
            print("LLM_JUDGE: 非皮肉")
            return "4"

def label_data(file_path, last_index, with_LLM):
    """データにラベルを付ける関数
    Args:
        file_path (str): ラベル付けするコーパスCSVの絶対パス
        last_index (int): 最後にラベル付けしたデータのインデックス
        with_LLM (bool): LLMを使うかどうか
    Returns:
        pandas.DataFrame: ラベル付けされたデータ
        index (int): 最後にラベル付けしたデータのインデックス
    """
    data = load_csv(file_path)  # CSVファイルを読み込む
    if data is None:
        return None, last_index
    if 'label' not in data.columns:  # label列が存在しない場合
        data['label'] = None  # label列を追加

    # start_indexから最終行までのデータ , iterrows()で行を取得 => (index, row)でindexと行を取得
    for index, row in data.iloc[last_index:].iterrows():  # データフレームの行を順番に処理

        # notna : NaN -> False
        if pd.notna(row['label']):  # labelがNaNでない場合
            continue  # 次の行へ
        print(f"index: {index}")  # インデックスを表示
        print(row["Utterance"].replace("\n", " "), end=" | ")
        print(row["Response"].replace("\n", " "))  # UtteranceとResponseを表示

        # 自動でのラベリング
        if with_LLM:
            label = labeling_auto(row)
            if label == "0":  # エラーが発生した場合
                is_break = labeling_manual(data, index)  # 手動ラベリング
                if is_break:
                    break
            elif label == "1" or label == "4":
                a_d = allow_or_deny()
                if a_d is not None:
                    if a_d:  # aが入力された場合
                        data.at[index, 'label'] = int(label)  # labelを更新
                    else:  # dが入力された場合
                        # label反転して更新
                        data.at[index, 'label'] = 4 if label == "1" else 1
                else:  # qが入力された場合
                    break
            else:
                print("error")
                is_break = labeling_manual(data, index)
                if is_break:
                    break
        else:
            is_break = labeling_manual(data, index)
            if is_break:
                break

    return data, index  # ラベル付けされたデータを返す


def save_csv(data, output_file_path):
    """データをCSVファイルに保存する関数
    Args:
        data (pandas.DataFrame): 保存するデータ
        output_file_path (str): CSVファイルのパス
    """
    try:
        data.to_csv(output_file_path, index=False)  # CSVファイルに保存
        print(f"データが{output_file_path}に保存されました。")  # 保存成功メッセージ
    except Exception as e:
        print(f"データの保存中にエラーが発生しました: {e}")  # エラーが発生した場合


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

    Absolute_path = os.path.dirname(__file__)  # 実行ファイルのディレクトリパス
    Work_path = os.path.join(Absolute_path, "work")  # 作業ディレクトリのパス

    # ディレクトリ内の候補(数字からはじまるディレクトリをリストアップ)
    all_items = os.listdir(Work_path)  # work/ディレクトリ内のファイルとディレクトリを取得
    directories = [d for d in all_items if os.path.isdir(
        os.path.join(Work_path, d))]  # ディレクトリのみを抽出
    # 数字から始まるディレクトリのみを抽出
    numeric_dirs = [d for d in directories if d[0].isdigit()]

    # ディレクトリの選択
    print(numeric_dirs)  # ディレクトリのリストを表示
    selected_dir = str(input("ディレクトリを選択してください: "))  # ディレクトリを選択
    selected_dir_path = os.path.join(Work_path, selected_dir)  # 選択されたディレクトリのパス

    # 進捗を確認
    # progress.jsonの存在を確認
    global progress_file_absolutepath
    progress_file_absolutepath = os.path.join(
        selected_dir_path, "progress.json")
    print(progress_file_absolutepath)
    selected_corpus, last_index = load_progress(
        progress_file_absolutepath)  # 進捗を読み込む

    # selected_corpusがNoneの場合の処理を追加
    if selected_corpus is None:
        # ファイルの選択
        all_files = os.listdir(selected_dir_path)
        csv_files = [f for f in all_files if f.endswith(
            ".csv")]  # CSVファイルのみを抽出
        print(csv_files)
        selected_file = str(input("ファイルを選択してください: "))
        last_index = 0
        create_progress_file(progress_file_absolutepath,
                             selected_file, last_index)
    else:
        selected_file = selected_corpus

    file_path = os.path.join(selected_dir_path, selected_file)  # これはコーパスの絶対パス
    print(f"選択されたファイル: {file_path}")

    while True:
        with_LLM = str(input("LLMを使いますか？ (y/n): "))
        if with_LLM == "y":
            with_LLM = True
            break
        elif with_LLM == "n":
            with_LLM = False
            break
        else:
            print("yまたはnを入力してください")

    if os.path.exists(file_path):
        data, index = label_data(file_path, last_index, with_LLM)  # ラベル付け
        if data is not None:
            output_file_path = os.path.join(
                selected_dir_path, file_path)  # 出力ファイルのパス
            save_csv(data, output_file_path)  # CSVファイルに保存
            save_progress(
                int(index)-1, progress_file_absolutepath, selected_file)
    else:
        print(f"ファイルが存在しません: {file_path}")


if __name__ == "__main__":
    main()
