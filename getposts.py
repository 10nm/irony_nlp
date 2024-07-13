### Mastodonのpublicタイムラインを取得、整形してから1000{unit}ポストごとにcsv(posts.csv)保存(追記)、
### 特定語句{check_word}を含むポストを抽出して、別で保存({check_word}.csv)する

### 環境変数 : Mastodonインスタンス側で取得するもの
### CID: client_id
### CIS: client_secret
### ACS: access_token

### BASEURL: api_base_url : Mastodonインスタンスのドメイン https:// **
### CHECKWORD: 検出したい語句

from mastodon import Mastodon, StreamListener
import os
import re
import csv
import time
from tqdm import tqdm

global unit
# 保存する単位(nポスト/回)
unit = 1000

global check_word
check_word = str(os.environ.get("CHECKWORD"))

def save_csv(posts, n):
    if n == 0:
        with open('csv/posts.csv', 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            for post in posts:
                writer.writerow([post])
    else:
        with open(f'csv/{check_word}.csv', 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([posts])

def remove(text):
    # RM htmlタグ
    html_tag = re.compile('<.*?>')
    text = re.sub(html_tag, '', text)
    # RM URL
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = re.sub(url_pattern, '', text)
    # RM EMOJI
    emoji_tag = re.compile(':[a-zA-Z]{5}:')
    text = re.sub(emoji_tag, '', text)
    # RM 改行
    text = text.replace('\n', ' ')
    # RM シングルクォート, ダブルクォート
    text = text.replace('\'', '')
    text = text.replace('\"', '')
    # RP \u3000 to space
    text = text.replace('\u3000', ' ')

    return text

class MyStreamListener(StreamListener):
    def __init__(self):
        self.posts = []
        self.posts_count = 0  
        self.all_posts_count = 0
        self.start_time = time.time() 
        self.csv_row_count = 0
        self.tqdm_bar = tqdm(total=unit, desc=f"{unit}Post", leave=True)
        
        with open('csv/posts.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            row_count = sum(1 for row in reader)
            self.csv_row_count = row_count

    def on_update(self, status):
        post = remove(status['content'])
        self.posts_count += 1
        self.all_posts_count += 1
        posts_per_sec = self.all_posts_count / (time.time() - self.start_time)
        if check_word in post:
            print(f"\n"+"\"{check_word}\"を含むポスト:"+str(post)+"\n")
            save_csv(post, 1)
       
        if self.posts_count < int(unit):
            self.posts.append(post)
            self.tqdm_bar.set_postfix({"PPS": f"{posts_per_sec:.2f}", "All Posts": self.csv_row_count + self.all_posts_count})
            self.tqdm_bar.update(1)

        else:
            print("")
            self.posts.append(post)
            self.posts_count = 0
            print(self.posts)
            print('-save csv-')
            save_csv(self.posts, 0)
            self.posts = []
            self.tqdm_bar.reset()
            
client_id=os.environ.get("CID")
client_secret=os.environ.get("CIS")
access_token=os.environ.get("ACS")
api_base_url=os.environ.get("BASEURL")

api = Mastodon(
    api_base_url=api_base_url,
    client_id=client_id,
    client_secret=client_secret,
    access_token=access_token
)

if __name__ == '__main__':
    listener = MyStreamListener()  
    api.stream_public(listener)
