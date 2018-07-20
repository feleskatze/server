import requests
from bs4 import BeautifulSoup
import xmltodict

def law_search(text):
    # textは'法令'でもらう
    text_set = text.split()
    command = text_set[0]
    if command == '#法令':
        lawname = text_set[1]
        article = text_set[2]
        RequestURL = 'http://elaws.e-gov.go.jp/api/1/articles;lawNum=' + lawname + ';article=' +article
        xml = requests.get(RequestURL)
        resdata = xmltodict.parse(xml.content.decode('utf-8', 'replace'))
        if resdata['DataRoot']['Result']['Code'] != '0':
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='取得できませんでした\n' + lawname + '\n' + article)
                )
            return 'OK'
        data = resdata['DataRoot']['ApplData']
        ReturnText = []
        ReturnText.append(data['LawNum'])

        ReturnText.append(data['LawContents']['Article']['ArticleTitle'])
        ReturnText.append(data['LawContents']['Article']['ArticleCaption'])
        for para in data['LawContents']['Article']['Paragraph']:
            te = para['@Num'] + ' ' + para['ParagraphSentence']['Sentence']['#text']
            ReturnText.append(te)
        return '\n'.join(ReturnText)

    elif command == '#法令番号':
        lawname = text_set[1]
        RequestURL = 'http://elaws.e-gov.go.jp/search/elawsSearch/elaws_search/lsg0100/search?searchType=2&searchLawName=' +lawname+'&abbreviationFlg=true'
        html = requests.get(RequestURL)
        soup = BeautifulSoup(html.content, "html.parser")
        items = soup.find_all('table', class_='table table-striped table-condensed table-bordered')

        ReturnText = ''
        count = 0
        for item in items:
            item_res = item.find_all('td')
            del item_res[:3]
            for item_one in item_res:
                if count % 3 == 0:
                    pass
                elif count % 3 == 1:
                    ReturnText = ReturnText + item_one.text
                elif count % 3 == 2:
                    ReturnText = ReturnText +': '+ item_one.text + '\n'
                count += 1
        return ReturnText
