# DSP 2020 Fall Final Project 

組員：b07902016 林義閔、b07902114 陳柏衡

## Usage

資料預處理，將會產生 `words.txt`、`documents.txt`

```
$ make preprocess
```

建立 W 矩陣，會產生 `W.npy`

```
$ make matrix
```

建立詞、新聞分群，結果分別為 `word_cluster.txt`、`document_cluster.txt`

```
$ make cluster
```