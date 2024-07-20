### Mastodonのpublicタイムラインを取得、整形してから1000{unit}ポストごとにcsv(posts.csv)保存(追記)、
### 特定語句{check_word}を含むポストを抽出して、別で保存({check_word}.csv)する

### 環境変数 : Mastodonインスタンス側で取得するもの
### CID: client_id
### CIS: client_secret
### ACS: access_token

### BASEURL: api_base_url : Mastodonインスタンスのドメイン https:// **
### CHECKWORD: 検出したい語句

from apscheduler.schedulers.background import BackgroundScheduler
from mastodon import Mastodon, StreamListener , MastodonNetworkError
import os
import re
import csv
import time
import datetime
import pandas
from tqdm import tqdm
import logging

global check_word
check_word = str(os.environ.get("CHECKWORD"))

#file paths
posts_path = 'logs/csv/posts.csv'
checkword_path = f'logs/csv/{check_word}.csv'
pps_path = 'logs/csv/posts_per_sec.csv'
error_log_path = 'logs/error.log'

global unit
# 保存する単位(nポスト/回)
unit = 50

scheduler = BackgroundScheduler()

global df_posts_per_sec
global posts_per_sec
df_posts_per_sec = pandas.DataFrame(columns=['Timestamp', 'PostsPerSec'])
df_posts_per_sec = pandas.read_csv(pps_path)
posts_per_sec = 0

global session_posts_count
session_posts_count = 0

# ログの設定
logging.basicConfig(filename='error.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s %(message)s')

# 記録インターバル(min)
interval = 5

def listen():
    listener = MyStreamListener()  
    api.stream_public(listener)


def savePPS():
    global df_posts_per_sec
    global posts_per_sec
    global session_posts_count
    print(session_posts_count)
    now = datetime.datetime.now()
    posts_per_sec = session_posts_count / (interval*60)
    new_row = {'Timestamp': now, 'PostsPerSec': posts_per_sec}
    df_posts_per_sec = pandas.concat([df_posts_per_sec, pandas.DataFrame([new_row])], ignore_index=True)
    df_posts_per_sec.to_csv(pps_path, index=False)
    session_posts_count = 0

def save_csv(posts, n):
    if n == 0:
        with open(posts_path, 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            for post in posts:
                writer.writerow([post])
    else:
        with open(checkword_path, 'a', encoding='utf-8') as f:
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
        self.session_count = 0
        self.csv_row_count = 0
        self.tqdm_bar = tqdm(total=unit, desc=f"session:{self.session_count}", leave=True)
        
        with open(posts_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            row_count = sum(1 for row in reader)
            self.csv_row_count = row_count

    def on_update(self, status):
        global posts_per_sec
        global session_posts_count
        session_posts_count += 1

        post = remove(status['content'])

        if check_word in post:
            print(f"\n"+"{check_word}を含むポスト:"+str(post)+"\n")
            save_csv(post, 1)
       
        if self.posts_count < int(unit):
            self.posts.append(post)
            self.tqdm_bar.set_postfix({"PPS": f"{posts_per_sec:.2f}", "All Posts": self.csv_row_count + self.posts_count + unit*self.session_count})
            self.tqdm_bar.update(1)
            self.posts_count += 1

        else:
            self.session_count += 1
            print("")
            self.posts.append(post)
            self.posts_count = 0
            print(self.posts)
            print('-save csv-')
            save_csv(self.posts, 0)
            self.posts = []
            self.tqdm_bar.reset()
            self.tqdm_bar = tqdm(total=unit, desc=f"session:{self.session_count}", leave=True)
            print(datetime.datetime.now())

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

scheduler.add_job(savePPS, 'interval', minutes=interval)
print('Scheduler started...')


if __name__ == '__main__':
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass

    while True:
        try:
            listen()
        except Exception as e:
            error_message = f"{datetime.now()} - Error: {str(e)}"
            logging.error(error_message)
            print(error_message)
            time.sleep(10)
            continue