import cohere
import os
import re
import time
import csv

api_key = os.getenv("CO_API_KEY")

co = cohere.Client(
    api_key=api_key,
)

# csvfile = "../irony_dataset.csv"
csvfile = "../test.csv"
def load_csv(file):
    not_ironic_texts = []
    with open(file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        counter = 0
        for row in reader:
            if counter > 10:
                break
            if int(row['label']) == 1:
                counter += 1
                not_ironic_texts.append(row['text'])
        print(f"Loaded {counter} texts")

    print(not_ironic_texts)
    return not_ironic_texts

file = "gen_ironic_texts.txt"

def saveTXT(file, texts):
    with open(file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text)
            f.write("\n")

def conv_ironic_text(texts):
    ironic = []
    for NotIronicText in texts:
        prompt = f"""
        与えられた日本語の文章を皮肉表現に変換してください。皮肉とは、相手を非難することを目的とした、婉曲的な表現です。
        推論は以下の手順従って、必ず1-4番の段階的推論ステップを出力しながら行ってください。
        結果は <ironic> </ironic> タグで囲んで、一番最後に出力してください。

        1. 感情分析・処理:
            - 文章の極性を解析
            - 文章がポジティブまたはニュートラルである場合のみ、ネガティブな内容に変換したものを、一度生成

        2. 文解析:
            - 話者の主張と感情を特定
            - 文脈における期待値を把握

        3. 対比要素の特定:
            - 現実の状況
            - 理想や期待される状況
            両者のギャップを皮肉の対象とする

        4. 皮肉表現の構築:
            - 表面上の賞賛/感心/同意
            - 間接的な批判/不満/疑問
            - 簡潔にする

        本番:
        入力: {NotIronicText}
        """

        chat = co.chat(
            message=prompt,
            model="command-r-plus"
        )
        
        chatbot_message = chat.chat_history[-1].message
        print(chatbot_message)
        pattern = r"<ironic>(.*?)</ironic>"
        matches = re.findall(pattern, chatbot_message, re.DOTALL)

        if matches:
            # 最後のマッチを取得
            last_ironic_text = matches[-1].strip()
            ironic.append(last_ironic_text)
        else:
            print("皮肉表現が見つかりませんでした。")
        time.sleep(3)
    return ironic

NotIronicTexts = load_csv(csvfile)
texts = conv_ironic_text(NotIronicTexts)
saveTXT(file, texts)
