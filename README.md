# BirdCLEF 2021 - Birdcall Identification
### Identify bird calls in soundscape recordings

## 締め切り
May 31, 2021 - Final submission deadline.

## 課題
意味のある異なる文脈で作成されたトレーニングデータを用いて、長時間の録音で397クラスのどの鳥が鳴いているかを特定すること
### 新しく追加されたデータ
- 新しい場所のサウンドスケープ
- 鳥種
- テストセットの録音に関するメタデータ
- トレーニングセットのサウンドスケープ

### ファイル
- train_short_audio
    - トレーニングデータの大部分は、xenocanto.orgのユーザーによって寛大にアップロードされた、個々の鳥の鳴き声の短い録音で構成
    - これらのファイルは、テストセットの音声に合わせて、必要に応じて32kHzにダウンサンプリングされ、oggフォーマットに変換
    - トレーニングデータには、ほぼすべての関連ファイルが含まれている
    - 意外と背景音がきれいな音が多い
    - 音量にばらつきあり
    - xenocanto.orgでこれ以上探すメリットはない
    - ratingは自身もしくはサイトにいる人がする。いつでも上書き可能で最後に評価された数値が反映されている
    - ファイル数は62874で、rating2.0以上は48886サンプル(足きりすると最小サンプルはstvhum2で4サンプルになる。しない場合はいくつか8サンプル, max500)
        - ratingが低い理由は音を聞いても分からない。その鳥以外の音が入っていても3.0以上のものもあれば、その鳥だけでそこまでノイズもないけど0.0なものもある。(本当にその鳥なのかは不明)
- train_soundscapes
    - テストセットとよく似たオーディオファイル
    - どれも10分程度の長さで、oggフォーマット
    - テストセットには、ここで紹介した2つの録音場所のサウンドスケープも含まれる
    - 訓練データに出現する鳥はかなり偏っている(47種類しか出現しない)
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
    - latitude & longitude：録音が行われた場所の座標
        - 鳥の種類によっては、鳴き声の「方言」がある場合があるので、学習データに地理的な多様性を持たせるとよい
    - date ：鳥の鳴き声には、警報音のように1年中鳴いているものもあれば、特定の季節に限定されるものもある
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
make apex
```
<details><summary>slurm用にヒアドキュメントを使用する場合</summary><div>

```
cd birdclef-2021/tools
sbatch << EOF
#!/bin/bash
make
make apex
EOF
```

</div></details>


## コンペの主題は何?

## 注意すること
- short_audioは一般人の投稿から作成されたデータなので、アノテーション規則が違う
- mask lossが前回の鳥カエルコンペで上位を占める (strong labelは0が指数関数的に増えるのでその対策に効果的)
- 背景で鳴いている鳥はnocall
    - 18003_CORはかなり静かなので、いつもノイズとされるレベルのrucwarが鳴いている音で判定されている
- 訓練データ2400個のチャンクのうちnocallは1529サンプル、nocall以外のサンプルは871個, 同一時間を含めるとのべ1183種
- 教師ラベルの前後でまだ鳴いているけどラベル付けされていないものがある
- 背景音はサンプルごとにかなり違う
- train_short_audioはすべてのサンプルで長さが違う。どこで鳴いているかの情報はないため、適当に切り出すと鳴き声がない可能性もある
- train_soundscape_labelsは教師として存在するサンプルが結構少ない (349/397クラスは1つもない)

## discussionで出てたこと
- np.uint8で学習して早い(20epoch 2hour)
- baggingでスコアめっちゃあがった
- 
## アイデア
- 

## 決定事項
- [x] trainにもvalidationにも、short_audioとsoundscapeの両方を用いる
    - ただし、soundscapeは各5秒数ごとに切る前提を置く。そしてバリデーションをする場合長い系列のfoldに合わせる。
    - nocallは排除して考える
- [x] b0, b7, b5, b3の優先順位でlrの探索をする
    - lrはバッチサイズ64で固定し2epochで10回検証データを計算しloss, f1スコアで比較する
    - lr: 1e-4, 3e-4, 8e-4, 2e-3, 5e-3, 1e-2
- [ ] data augmentation
    - ボリュームを0.8~1.2倍
    - mixup
    - specaug (default)
    - [ ] fold0のb0, 20epochで比較
        - なにもなし
        - 3つの比較をする
    - [ ] fold0のb7, 20epochで比較
        - なにもなし
        - 3つの比較をする
- [ ] 5foldの推論とポストプロセスを実行
    - oofはpredと答えが欲しいb7-mixup
- [ ] short audioに対してpseudo-labelして学習データを厳選してから再度学習
- [ ] wavenetはmixupの後
- [ ] https://github.com/Cadene/pretrained-models.pytorch#senet のse_resnext101_32x4dを変更する

### 実験結果からの気づき
- 最適なlr
    - b0 : 2e-3, b3 : 2e-3, b5 : 5e-3 , b7 : 2e-3
    - 2epoch時点でのベストの精度比較: b0 > b3 > b5 > b7
### 実験したモデルたち
- 
### 今の課題は何?
- 

<details><summary>kaggle日記</summary><div>

- 4/23(金)
    - 今日やったこと
        * リポジトリ作成&コンペの理解
    - 次回やること
        * ubuntu&centos両方サーバー上で動くmakefileに修正
        * 使えるデータ一覧を確認
- 4/24(土)
    - 今日やったこと
        * ubuntu&centos両方サーバー上で動くmakefileに修正
        * 使えるデータ一覧を確認
        * 音聞く
        * アライさんのコードkernelで動かす
    - 次回やること
        * 手元環境でのEDAと1st subの作成
</div></details>
