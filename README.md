# Irony_NLP

作業記録・詳細情報: [scarpbox NLP_irony](https://scrapbox.io/NLP-irony/)

## コーパスの収集

### Mastodonのリレーを活用して収集
※収集はしたものの、今回は不採用

Scrapbox: [Mastodonをコーパス収集に活用する](https://scrapbox.io/NLP-irony/03_Mastodon%E3%82%92%E3%82%B3%E3%83%BC%E3%83%91%E3%82%B9%E5%8F%8E%E9%9B%86%E3%81%AB%E6%B4%BB%E7%94%A8%E3%81%99%E3%82%8B)  
Code: [mastodon_corpus_collection](./mastodon_corpus_collection/)
### 概要
- Mastodonの公開タイムラインから全ポストの保存(csv保存 , API利用)
- 特定語句を含むポストを抽出、保存(csv保存)
- ポスト流量の計測、記録(csv保存)

#### 使い方
1. 環境変数の設定

2. 実行ディレクトリに移動して実行
```
python3 getposts.py
```

### 仕様
#### プログラム
`getposts.py` : ポストの保存、キーワード抽出、ポスト流量の計測  
`readPPS.py` : ポスト流量の読み込み、グラフ化

#### ライブラリ(動作確認済み)



#### 環境変数

`CID`: MastodonのClient ID  
`CIS`: MastodonのClient Secret  
`ACS`: MastodonのAccess Token  
`BASEURL`: インスタンスのURL  
`CHECKWORD`: 抽出したいキーワード  

#### 変数

`unit` : 全ポストの保存単位(ポスト)  
`interval` : ポスト流量の計測間隔(秒)

#### 保存パス
全ポスト -> `./logs/csv/posts.csv`  
キーワード抽出 -> `./logs/csv/{CHECKWORD}.csv`  
ポスト流量({interval}分間隔) -> `./logs/csv/post_per_sec.csv`

#### 改善点
- ~~ポスト流量を平均化~~
- ~~キーワード抽出項目をリストから読み込む~~

### おーぷん2ちゃんねる対話コーパス[1]を活用した皮肉表現コーパス収集

Scrapbox: [おーぷん2ちゃんねる対話コーパス[1]を活用した皮肉表現コーパス収集](https://scrapbox.io/NLP-irony/%E3%81%8A%E3%83%BC%E3%81%B7%E3%82%932%E3%81%A1%E3%82%83%E3%82%93%E3%81%AD%E3%82%8B%E5%AF%BE%E8%A9%B1%E3%82%B3%E3%83%BC%E3%83%91%E3%82%B9_1_%E3%82%92%E6%B4%BB%E7%94%A8%E3%81%97%E3%81%9F%E7%9A%AE%E8%82%89%E8%A1%A8%E7%8F%BE%E3%82%B3%E3%83%BC%E3%83%91%E3%82%B9%E5%8F%8E%E9%9B%86%E3%82%A2%E3%83%97%E3%83%AD%E3%83%BC%E3%83%81)  
Code: [2chan_corpus_collection/  read](./2chan_corpus_collection/),   [bert-train/  search](./bert-train/corpus/search.py)

#### 実装
read.py:  
各掲示板のコーパスを読み込み、クエリを用いて探索、ヒットしたものをcsvで保存

search.py:  
↑で保存したcsvを読み込み、種別ごとに分類、json形式で分けて再保存する

#### 結果
ヒットしたものは約450件、(厳密に)重複を削除した場合 約250件、クエリが不十分または収集方法に誤りがある？

#### データセット
[1]. 稲葉 通将. おーぷん2 ちゃんねる対話コーパスを用いた用例ベース対話システム. In 第87 回
言語・音声理解と対話処理研究会(第10 回対話システムシンポジウム), 人工知能学会研究会
資料 SIG-SLUD-B902-33, pages 129–132, 2019.

おーぷん2ちゃんねる対話コーパス  
https://github.com/1never/open2ch-dialogue-corpus

## BERTの分類モデルをファインチューニング
...