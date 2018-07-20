from bs4 import BeautifulSoup
import urllib.request
import jaconv

def search(text):
    RequestURL = 'http://usjinfo.com/wait/realtime.php'
    html = urllib.request.urlopen(RequestURL).read()
    if text.find('#USJ') == 0 and len(text) > 5:
        att = text[4:].split()
    elif text.find('#USJ') == 0 and len(text) < 5:
        att = False
    soup = BeautifulSoup(html, "html.parser")
    ReturnText = []
    attractions = []
    item = {}
    li = soup.find_all('ul', class_='list')[1].find_all('li')
    for tag in li:
        if tag.text.find('現在')  != -1:
            pass
        elif tag.text.find('分') != -1 or tag.text.find('休止') != -1 or tag.text.find('終了') != -1:
            item.update({'info': tag.text.replace(' ','').replace('\n', '')})
            attractions.append(item)
        else:
            item = {'title': jaconv.h2z(tag.text.replace(' ','').replace('\n', ''))}
    for se in attractions:
        if att is not False:
            if se['title'].find(att[0]) != -1:
                ReturnText.append(se['title'] + ': ' + se['info'])
        elif att is False:
            ReturnText.append(se['title'] + ': ' + se['info'])
    if not ReturnText:
        ReturnText.append('見つかりませんでした。検索キーワードを見直すか、管理者に問い合わせてください。')

    return '\n'.join(ReturnText)