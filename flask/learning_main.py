# -*- coding:utf-8 -*-
import sys
import os
import cv2
import random
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

# 識別ラベルの数(今回は4つ)
NUM_CLASSES = 4
# 学習する時の画像のサイズ(px)
IMAGE_SIZE = 28
# 画像の次元数(28px*28px*3(カラー)
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

# Flagはデフォルト値やヘルプ画面の説明文を定数っぽく登録できるTensorFlowの組み込み関数
flags = tf.app.flags
FLAGS = flags.FLAGS
# 学習用データ
flags.DEFINE_string('train', './data/train/train_data.txt', 'File name of train data.')
# 検証用データ
flags.DEFINE_string('test', './data/test/test_data.txt', 'File name of test data.')
# TensorFlowのデータ保存先フォルダ
flags.DEFINE_string('train_dir', './data', 'Directory to put the training data.')
# 学習訓練の試行回数
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
# 1回の学習で何枚の画像を使うか
flags.DEFINE_integer('batch_size', 20, 'Batch size Must divide evenly into the dataset sizes.')
# 学習率、小さすぎると学習が進まないし、大きすぎても誤差が収束しなかったり発散したりして駄目とか
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')


#-------
#サンプルではここに学習モデル構築コードが入ってる

# AIの学習モデル部分(ニューラルネットワーク)
# images_placeholder:画像のplaceholder, keep_prob:dropout率のplace_holderが引数となり入力画像に対して各ラベルの確率を出力して返す
def inference(images_placeholder, keep_prob):
    
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 畳み込み層を作成する
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # ブーリング層を作成する
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # ベクトル形式で入力されてきた画像データを28px*28pxの画像に戻す
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    # 畳み込み層第1レイヤーを作成
    with tf.name_scope('conv1') as scope:
        # 引数は[width, height, inpu, filters] 5px*5pxの範囲で画像をフィルター 32個の特徴を検出
        W_conv1 = weight_variable([5, 5, 3, 32])
        # バイアスの数値を代入
        b_conv1 = bias_variable([32])
        # 特徴として検出した有用そうな部分は残し、特徴として使えない部分は0として、特徴として扱わないようにする
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # ブーリング層1の作成
    # 2*2の大きさの枠を作成、その枠内の特徴を1*1の大きさに変換、枠をスライドさせて全体を圧縮、特徴をまとめる
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    # 畳み込み第2レイヤーの作成
    with tf.name_scope('conv2') as scope:
        # 第一レイヤーでの出力を第2レイヤー入力にしてもう一度フィルタリング、5px*5pxの範囲で画像をフィルター32個の特徴を入力64個の特徴を検出
        W_conv2 = weight_variable([5, 5, 32, 64])
        # バイアスの数値を代入
        b_conv2 = bias_variable([64])
        # 検出した特徴の整理
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # ブーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        # 画像の解析の結果をベクトルへ変換
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        # 第一、第二と同じく、検出した特徴を活性化
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定　訓練用データだけに最適化してしまい、実際に使えないようなAIになってしまう「過学習」を防止
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全結合層2の作成(読み出しレイヤー)
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
        
    # ソフトマックス関数による正規化　ここまでのニューラルネットワークの出力を各ラベルの確率へ変換する
    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率(のようなもの)を返す。算出された各ラベルの確率を全て足すと1になる
    return y_conv


# 予測結果と正解にどれくらい「誤差」があったかを算出する
# logitsは計算結果: float - [batch_size, NUM_CLASSE]
# labelsは正解ラベル: int32 -[batch_size, NUM_CLASSE]
def loss(logits, labels):
    # 交差エントロピー計算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
     # TensorBoardで表示するよう指定
    tf.summary.scalar("cross_entropy", cross_entropy)
    # 誤差の率の値(cross_entropy)を返す
    return cross_entropy

# 誤差(loss)を元に誤差逆伝播を用いて設計した学習モデルを訓練する
# 裏側何が起きているのかよくわかってないが、学習モデルの各層の重み(w)などを
# 誤差を元に最適化してパラメーターを調整しているという理解(?)
# (誤差逆伝播は「人工知能は人間を超えるか」書籍の説明が神)
def training(loss, learning_rate):
    #この関数がその当たりの全てをやってくれる様
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

# inferenceで学習モデルが出した予測結果の正解率を算出する
def accuracy(logits, labels):
    # 予測ラベルと正解ラベルが等しいか比べる。同じ値であればTrueが返される
    # argmaxは配列の中で一番値の大きい箇所のindex(=一番正解だと思われるラベルの番号)を返す
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    # booleanのcorrect_predictionをfloatに直して正解率の算出
    # false:0,true:1に変換して計算する
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # TensorBoardで表示する様設定
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

if __name__ == '__main__':
    # 学習用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
    # ファイルを開く
    f = open(FLAGS.train, 'r')
    # データを入れる配列
    train_image = []
    train_label = []
    for line in f:
        # 改行を除いてスペース区切りにする
        line = line.rstrip()
        l = line.split()
        # データを読み込んで28x28に縮小
        img = cv2.imread(l[0])
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        # 一列にした後、0-1のfloat値にする
        train_image.append(img.flatten().astype(np.float32)/255.0)
        # ラベルを1-of-k方式で用意する
        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        train_label.append(tmp)
    # numpy形式に変換
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)
    f.close()

    # 同じく検証用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
    f = open(FLAGS.test, 'r')
    test_image = []
    test_label = []
    for line in f:
        line = line.rstrip()
        l = line.split()
        img = cv2.imread(l[0])
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        test_image.append(img.flatten().astype(np.float32)/255.0)
        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        test_label.append(tmp)
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    f.close()

    #TensorBoardのグラフに出力するスコープを指定
    with tf.Graph().as_default():
        # 画像を入れるためのTensor(28*28*3(IMAGE_PIXELS)次元の画像が任意の枚数(None)分はいる)
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        # ラベルを入れるためのTensor(3(NUM_CLASSES)次元のラベルが任意の枚数(None)分入る)
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder("float")

        # inference()を呼び出してモデルを作る
        logits = inference(images_placeholder, keep_prob)
        # loss()を呼び出して損失を計算
        loss_value = loss(logits, labels_placeholder)
        # training()を呼び出して訓練して学習モデルのパラメーターを調整する
        train_op = training(loss_value, FLAGS.learning_rate)
        # 精度の計算
        acc = accuracy(logits, labels_placeholder)

        # 保存の準備
        saver = tf.train.Saver()
        cwd = os.getcwd()
        # Sessionの作成(TensorFlowの計算は絶対Sessionの中でやらなきゃだめ)
        sess = tf.Session()
        # 変数の初期化(Sessionを開始したらまず初期化)
        sess.run(tf.initialize_all_variables())
        # TensorBoard表示の設定(TensorBoardの宣言的な?)
        summary_op = tf.summary.merge_all()
        # train_dirでTensorBoardログを出力するpathを指定
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph_def)

        # 実際にmax_stepの回数だけ訓練の実行していく
        for step in range(FLAGS.max_steps):
            for i in range(int(len(train_image)/FLAGS.batch_size)):
                # batch_size分の画像に対して訓練の実行
                batch = FLAGS.batch_size*i
                # feed_dictでplaceholderに入れるデータを指定する
                sess.run(train_op, feed_dict={
                    images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                    labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
                    keep_prob: 0.5})

            # 1step終わるたびに精度を計算する
            train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            print("step %d, training accuracy %g"%(step, train_accuracy))

            # 1step終わるたびにTensorBoardに表示する値を追加する
            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

    # 訓練が終了したらテストデータに対する精度を表示する
    print("test accuracy %g"%sess.run(acc, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0}))

    # データを学習して最終的に出来上がったモデルを保存
    # "model.ckpt"は出力されるファイル名
    save_path = saver.save(sess, cwd+"./model.ckpt")
