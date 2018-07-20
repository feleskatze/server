#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
import tensorflow as tf
import os
import random
import learning_main

# OpenCVのデフォルトの顔の分類器のpath
cascade_path = '/home/feleskatze/www/flask/haarcascade_frontalface_alt2.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

# 識別ラベルと各ラベル番号に対応する名前
HUMAN_NAMES = {
  0: u"花澤香菜",
  1: u"佐倉綾音",
  2: u"茅野愛衣",
  3: u"悠木碧"
}


def detect(img, ckpt_path):
    image = []

    img = cv2.resize(img, (28, 28))
    image.append(img.flatten().astype(np.float32)/255.0)
    image = np.asarray(image)
    tf.reset_default_graph()
    logits = learning_main.inference(image, 1.0)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, os.getcwd() + ckpt_path)
        softmax = logits.eval()
        result = softmax[0]
        rates = [round(n * 100.0, 1) for n in result]
        results = []
        for index, rate in enumerate(rates):
            name = HUMAN_NAMES[index]
            results.append({
                'label': index,
                'name': name,
                'rate': rate
                })
    return results





#指定した画像(img_path)を学習結果(ckpt_path)を用いて判定する
def evaluation(img_path, ckpt_path):
    humans = []
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 3)
    if len(faces) > 0:
        for rect in faces:
            if abs(rect[2] - rect[3]) > 3:
                continue
            ramdom_str = str(random.random())
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]

            dst = image[y:y+h, x:x+w]
            dst_file_name = ramdom_str + '.jpg'
            dst_file_path =  os.path.join('/home/feleskatze/www/flask/tmp', dst_file_name)

            cv2.imwrite(dst_file_path ,dst)
            result = detect(dst, ckpt_path)
            human = {}
            human['x'] = x
            human['y'] = y
            human['width'] = w
            human['height'] = h
            human['dst_file_path'] = '/tmp/' + dst_file_name
            human['rank'] = sorted(result, key=lambda x: x['rate'], reverse=True)
            humans.append(human)

            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0,0,255), thickness=2)
    else:
        return None
    cv2.imwrite(img_path, image)
    return humans if humans != [] else None

# コマンドラインからのテスト用
if __name__ == '__main__':
  evaluation('testimage.jpg', './model.ckpt')
