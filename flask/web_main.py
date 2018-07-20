# -*- coding: utf-8 -*-
import sys, os
import urllib.request
import json
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import requests
import xmltodict
import yaml

import tensorflow as tf
import multiprocessing as mp

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from werkzeug import secure_filename

# LINE Message API
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    LineBotApiError, InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    TemplateSendMessage, ConfirmTemplate, MessageAction,
    ButtonsTemplate, ImageCarouselTemplate, ImageCarouselColumn, URIAction,
    PostbackAction, DatetimePickerAction,
    CarouselTemplate, CarouselColumn, PostbackEvent,
    StickerMessage, StickerSendMessage, LocationMessage, LocationSendMessage,
    ImageMessage, VideoMessage, AudioMessage, FileMessage,
    UnfollowEvent, FollowEvent, JoinEvent, LeaveEvent, BeaconEvent,
    FlexSendMessage, BubbleContainer, ImageComponent, BoxComponent,
    TextComponent, SpacerComponent, IconComponent, ButtonComponent,
    SeparatorComponent,
)

# 自作をインポート
import eval
from module import law
from module import weather
from module import biblio
from module import USJ
from module import TD



# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)
app = Flask(__name__, static_folder='tmp')
# 投稿画像の保存先
FULL_PATH = '/home/feleskatze/www/flask/'
UPLOAD_FOLDER = 'tmp/'




# LINE API KEY
# 環境変数から取得
with open('/home/feleskatze/www/key.yaml') as file:
    API_KEY = yaml.load(file)

line_bot_api = LineBotApi(API_KEY['channel_access_token'])
handler = WebhookHandler(API_KEY['channel_secret'])

# ルーティング。/にアクセス時
@app.route('/')
def index():
    return render_template('index.html')

# 画像投稿時のアクション
@app.route('/post', methods=['GET','POST'])
def post():
    if request.method == 'POST':
        if not request.files['file'].filename == u'':
            # アップロードされたファイルを保存
            f = request.files['file']
            img_path = FULL_PATH + UPLOAD_FOLDER + f.filename
            f.save(img_path)
            # eval.pyへアップロードされた画像を渡す
            result = eval.evaluation(img_path, '/home/feleskatze/www/flask/model.ckpt')
        if result == None:
            return redirect(url_for('index'))
        else:
            KANA = False
            rect = [[],[],[],[]]
            for human in result:
                rect[0].append(human['x'])
                rect[1].append(human['y'])
                rect[2].append(human['width'])
                rect[3].append(human['height'])
                if human['rank'][0]['rate'] >= 90:
                    KANA = True
            return render_template('index.html', img_path='./tmp/' + f.filename, rect=rect, result=result, KANA=KANA)

    else:
        # エラーなどでリダイレクトしたい場合
        return redirect(url_for('index'))




#====================
# LINE MessageAPI
#====================


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'




# テキストメッセージの時
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text = event.message.text

    if text == '#ヘルプ':
        ReturnText = 'お天気通知: #天気東京 #天気横浜 #天気神戸\n天気予報詳細と天気・気温を表示\n情報元:https://darksky.net/forecast/'
        ReturnText = ReturnText + '\n\nTD待ち時間通知: #TDL #TDS\nTDLとTDSの待ち時間を通知．改行後にアトラクション名を指定可能\n情報元:http://tokyodisneyresort.info'
        ReturnText = ReturnText + '\n\nUSJ待ち時間通知: #USJ\nUSJの待ち時間通知．改行後にアトラクション名を指定可能\n情報元:http://usjinfo.com/wait/realtime.php'
        ReturnText = ReturnText + '\n\n書誌情報取得: #ISBN[ISBN]\n例: #ISBN 9784041025475 #ISBN9784041025475\n 情報元:国立国会図書館サーチ(API)'
        ReturnText = ReturnText + '\n\n法令番号　#法令番号 [改行] [検索したい法令名] \n情報元:e-Gov法令検索 http://elaws.e-gov.go.jp/search/elawsSearch/elaws_search/lsg0100/'
        ReturnText = ReturnText + '\n\n法令　#法令 [改行] [検索したい法令番号] [改行] [検索したい条]\n情報元:法令API'
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text = ReturnText)
        )



    if text.find('#天気') == 0 and len(text) > 3:
        ReturnText = weather.weather(text)
        line_bot_api.reply_message(
            event.reply_token,[
            TextSendMessage(text=ReturnText[0]),
            TextSendMessage(text=ReturnText[1])
            ])
    
    # TDL TDS
    if text.find('#TDL') == 0 or text.find('#TDS') == 0:
        ReturnText = TD.search(text)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=ReturnText)
        )

    if text.find('#USJ') == 0:
        ReturnText = USJ.search(text)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=ReturnText)
        )
    
    if text.find('#ISBN') == 0 and len(text) > 4:
        ReturnText = biblio.isbn(text)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=ReturnText)
        )

# 法令
    if text.find('#法令') == 0 and len(text) > 3:
        ReturnText = law.law_search(text)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=ReturnText)
            )


if __name__ == '__main__':
    app.run(port='80')
