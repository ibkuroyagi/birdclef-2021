# BirdCLEF 2021 - Birdcall Identification
### Identify bird calls in soundscape recordings

## 締め切り
May 31, 2021 - Final submission deadline.

## 課題
意味のある異なる文脈で作成されたトレーニングデータを用いて、長時間の録音でどの鳥が鳴いているかを特定すること
### 追加されたデータ
- 新しい場所のサウンドスケープ
- 鳥種
- テストセットの録音に関するメタデータ
- トレーニングセットのサウンドスケープ

### ファイル
- train_short_audio
    - トレーニングデータの大部分は、xenocanto.orgのユーザーによって寛大にアップロードされた、個々の鳥の鳴き声の短い録音で構成
    - これらのファイルは、テストセットの音声に合わせて、必要に応じて32kHzにダウンサンプリングされ、oggフォーマットに変換
    - トレーニングデータには、ほぼすべての関連ファイルが含まれている
    - xenocanto.orgでこれ以上探すメリットはない
- train_soundscapes
    - テストセットとよく似たオーディオファイル
    - どれも10分程度の長さで、oggフォーマット
    - テストセットには、ここで紹介した2つの録音場所のサウンドスケープも含まれる
- test_soundscapes
    - ノートブックを提出すると、test_soundscapesディレクトリには、スコアリングに使用する約80個の音源が格納される
    - これらのファイルはおよそ10分で、oggオーディオフォーマット
    - ファイル名には録音された日付が含まれており、渡り鳥の識別に特に役立つ
    - 録音場所の名前とおおよその座標が記載されたテキストファイル
    - テストセットのサウンドスケープが録音された日付のセットが記載されたcsv
- test.csv 
    - 最初の3行のみがダウンロード可能
    - 完全なtest.csvは隠しテストセットにある
- train_soundscape_labels.csv
    - row_id : その行のIDコード
    - site : サイトID
    - seconds : 時間軸の最後の秒数
    - audio_id : オーディオファイルのIDコード
- train_metadata.csv
    - primary_label: 鳥の種類を表すコード。https://ebird.org/species/ にコードを追加することで、鳥類コードの詳細情報を確認できる
    - 例えば、アメリカクロウの場合は https://ebird.org/species/amecro
    - recodist：録音を提供したユーザー
    - 緯度と経度：録音が行われた場所の座標
        - 鳥の種類によっては、鳴き声の「方言」がある場合があるので、学習データに地理的な多様性を持たせるとよい
    - 日付：鳥の鳴き声には、警報音のように1年中鳴いているものもあれば、特定の季節に限定されるものもある
        - トレーニングデータには、時間的な多様性を求めるとよい
    - filename：関連するオーディオファイルの名前
- train_soundscape_labels.csv
    - row_id: 行のIDコード
    - site: サイトID
    - seconds: 時間枠を終了する秒数
    - audio_id: 音声ファイルのIDコード
    - birds: 5秒間に鳴いていた鳥の声をスペースで区切ったリスト。nocallというラベルは、鳴き声がなかったことを意味する

- sample_submission.csv 
    - 適切に作成されたサンプル投稿ファイル。最初の3行だけが公開される
    - row_id
    - birds: 5秒間のウィンドウ内に存在するすべての鳥の鳴き声をスペースで区切ったリスト。鳥の鳴き声がない場合は、nocallというラベルを使う。
## CV vs PL
スプレッドシートにCVとPLの関係を記録する
[https://docs.google.com/spreadsheets/d/1B4jWU7fQP6d8AIPqbTKUmyZApJLixgRrRH0r90F427Q/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1B4jWU7fQP6d8AIPqbTKUmyZApJLixgRrRH0r90F427Q/edit?usp=sharing)

## introduction
ホームディレクトリに.kaggleディレクトリが作成されている前提で作成します。 
ない場合は、こちら[https://www.currypurin.com/entry/2018/kaggle-api](https://www.currypurin.com/entry/2018/kaggle-api)を参照してください。
```
# リポジトリのクローン
git clone https://github.com/ibkuroyagi/birdclef-2021.git
# 環境構築
cd birdclef-2021/tools
make
```
<details><summary>slurm用にヒアドキュメントを使用する場合</summary><div>

```
cd birdclef-2021/tools
sbatch -c 4 -w million2 << EOF
#!/bin/bash
make
EOF
```

</div></details>


## コンペの主題は何?

## 注意すること
- 

## アイデア
- 

## 決定事項
- 
## 実験結果からの気づき
- 
### 実験したモデルたち
- 
### 今の課題は何?
- 

<details><summary>kaggle日記</summary><div>

- 4/23(金)
    - 今日やったこと
        * リポジトリ作成&コンペの理解
    - 次回やること
        * 手元環境でのEDAと1st subの作成

</div></details>
