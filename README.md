## はじめに
初めまして、Rollphesと言う者です。普段は原神で遊びまくってるただの社会人ニ年目です。<br>
このレポジトリは[テイワットマップ](https://act.hoyolab.com/ys/app/interactive-map/index.html)を自動でゲームと追尾させたいと思い2022/2に作った物になります。<br>
プログラミングは趣味でやってるだけで独学なので説明やコードに不備があるかと思いますが見なかったことにしてくださいｗ<br>
GPUを使ってないのもそのせいです(ただの勉強不足)ｗ<br>
もし問題点等あればissue等を用いて頂ければ幸いです。<br>
もちろんTwitter(ユーザー名とID一緒です)やDiscordでも対応できます。<br>

## 動作環境
・windows CPU:i7以上(負荷は大体原神の半分程度です)<br>
・Chrome<br>
・ディスプレイの比率が16:9(16:10でも恐らく動く)<br>
・原神をフルスクリーンでプレイしている事<br>
・原神をメインディスプレイでプレイしている事<br>
※1:自分の環境はこれなので別バージョンでの動作は保証しません。<br>
※2:実行時のメモリ消費量は約600MBです。

## 導入方法
[Releases](https://github.com/Rollphes/Sync-teyvat-map/releases)から最新バージョンをインストールするだけです。<br>
[最新バージョンはコチラ](https://github.com/Rollphes/Sync-teyvat-map/releases/latest)<br>

注意:WindowsによってPCが保護されましたという警告が出ます。詳細情報をクリックすれば実行ボタンが出てきますのでそちらから実行ください。<br>

尚、新バージョンが出た場合は自動で更新されます。


## 実行方法
起動するだけです。
## 仕組み(如何せん10ヶ月前のコードなのでほぼ忘れてます。)
1.スクリーンショットをt.pngとして保管<br>
2.datとして保管してあるマップデータ(setupスクリプト内でAKAZE特徴量と特徴記述を保管した者)とマッチング<br>
3.精度を高めるためマッチングした結果のソートを実施<br>
4.特にマッチング距離が短かった40setを抽出<br>
5.角度違いを考え、更にマッチングを実施<br>
6.抽出した2つのマッチングからベクトルを取り出し、テイワットマップの原点からのベクトルを算出<br>
7.WebSocketを用いてローカル通信でChromeに送信<br>
8.拡張機能が受信してURLのパラメータ部分を書き換える。<br>
9.パラメータを書き換えた事によりテイワットマップが動く

## 開発者向け
動作を軽くするために拡張子dat内にjson形式でKeyPointsとDescriptorsを保管しています。<br>
もしマップが拡張された際に自分でマップ画像`src/???.png`を弄って更新したい場合は`setup`スクリプトを実行してください。<br>
32GBメモリを実装しているPCでもほぼ全部持ってかれるので注意してください。<br>
尚、事前にsetupを実行した物を保管しているので更新しなくても動きます。