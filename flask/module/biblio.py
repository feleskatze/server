import urllib.request
import xml.etree.ElementTree as ET

def isbn(text):
    isbn = text[5:]
    RequestURL = "http://iss.ndl.go.jp/api/opensearch"
    RequestURL = RequestURL + "?isbn=" + isbn.replace(" ","")
    XMLdata = urllib.request.urlopen(RequestURL).read()
    items = ET.fromstring(XMLdata).getiterator("item")
    item_set = {}
    ReturnText = ''
    for item in items:
        item_set['書名'] = item.find("{http://purl.org/dc/elements/1.1/}title")
        item_set['巻次'] = item.find("{http://ndl.go.jp/dcndl/terms/}volume")
        item_set['著者'] = item.findall("{http://purl.org/dc/elements/1.1/}creator")
        item_set['シリーズ"'] = item.find("{http://ndl.go.jp/dcndl/terms/}seriesTitle")
        item_set['出版社'] = item.find("{http://purl.org/dc/elements/1.1/}publisher")
        item_set['出版年'] = item.find("{http://purl.org/dc/terms/}issued")
        for key, value in item_set.items():
            if type(value) is list and value is not None:
                ReturnText = ReturnText + key + ': '
                for item_cre in value:
                    ReturnText = ReturnText + item_cre.text.replace(',','') + ', '
                ReturnText = ReturnText + '\n'
            elif value is not None:
                ReturnText = ReturnText + '{}: {}'.format(key, value.text) + '\n'
        item_set.clear()
        ReturnText = ReturnText + '\n'
    return ReturnText