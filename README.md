# Mastodon Post Analyzer

## 概要
- Mastodonの公開タイムラインから全ポストの保存(csv保存 , API利用)
- 特定語句を含むポストを抽出、保存(csv保存)
- ポスト流量の計測、記録(csv保存)

### 使い方
1. 環境変数を設定
2. 実行ディレクトリに移動
```
mkdir csv
python3 getposts.py
```

## 仕様
### プログラム
`getposts.py` : ポストの保存、キーワード抽出、ポスト流量の計測  
`readPPS.py` : ポスト流量の読み込み、グラフ化

### ライブラリ(動作確認済み)
APScheduler==3.10.4  
Mastodon.py==1.8.1  
pandas==2.2.2  
tqdm==4.66.4


### 環境変数

`CID`: MastodonのClient ID  
`CIS`: MastodonのClient Secret  
`ACS`: MastodonのAccess Token  
`BASEURL`: インスタンスのURL  
`CHECKWORD`: 抽出したいキーワード  

### 変数

`unit` : 全ポストの保存単位(ポスト)  
`interval` : ポスト流量の計測間隔(秒)

### 保存パス
全ポスト -> `./csv/posts.csv`  
キーワード抽出 -> `./csv/{CHECKWORD}.csv`  
ポスト流量({interval}分間隔) -> `./csv/post_per_sec.csv`

### 予定
- ポスト流量を平均化
- キーワード抽出項目をリストから読み込む
