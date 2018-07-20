from bs4 import BeautifulSoup
import urllib.request
import jaconv

def search(text):
    if len(text) > 5:
        att = text[4:].split()
    elif  len(text) < 5:
        att = False
    if text.find('#TDS') == 0:
        RequestURL = 'http://tokyodisneyresort.info/smartPhone/realtime.php?park=sea&order=wait'
    elif text.find('#TDL') == 0:
        RequestURL = 'http://tokyodisneyresort.info/smartPhone/realtime.php?park=land&order=wait'        
    html = urllib.request.urlopen(RequestURL).read()
    soup = BeautifulSoup(html, "html.parser")
    li = soup.find_all("li")
    ReturnText = []
    attractions = []
    item = {}
    for tag in li:
        if tag.text.find('現在')  != -1:
            pass
        elif tag.text.find('更新') != -1 or tag.text.find('案内') != -1:
            item.update({'info': tag.text.replace(' ','').replace('\n', '')})
            attractions.append(item)
        elif tag.text.find('現在')  != -1:
            pass
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