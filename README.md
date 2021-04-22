# BirdCLEF 2021 - Birdcall Identification
### Identify bird calls in soundscape recordings

## 締め切り
May 31, 2021 - Final submission deadline.

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
