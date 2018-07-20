import json
import urllib.request
import yaml

def weather(text):
    with open('/home/feleskatze/www/key.yaml') as file:
        API_KEY = yaml.load(file)
    city_set= {'Yokohama': '35.445,139.6368', 'Tokyo': '35.719444, 139.707917', 'Kobe': '34.7264,135.2354'}
    if text[3:] == '横浜':
        city = city_set['Yokohama']
    elif text[3:] == '東京':
        city = city_set['Tokyo']
    elif text[3:] == '神戸':
        city = city_set['Kobe']
    else:
        return 'OK'
    RequestURL = 'https://api.darksky.net/forecast/' + API_KEY['FORECAST_DARKSKY_API_KEY'] + '/' + city +'?units=auto'
    data = json.load(urllib.request.urlopen(RequestURL))
    curr = data['currently']
    today = data['daily']['data'][0]
    curr_set = []
    today_set = []
    curr_set.append(text[3:] + 'の現在の天気')
    curr_set.append('天気概要: ' + curr['summary'])
    curr_set.append('天気: ' + curr['icon'])
    curr_set.append('気温: ' + str(curr['temperature']) + '℃')
    curr_set.append('体感気温: ' + str(curr['apparentTemperature']) + '℃')
    curr_set.append('湿度: ' + str(curr['humidity']*100) + '%')
    curr_set.append('気圧: ' + str(curr['pressure']) + 'hPa')
    today_set.append(text[3:] + 'の今日の天気')
    today_set.append('天気概要: ' + today['summary'])
    today_set.append('天気: ' + today['icon'])
    today_set.append('最高気温: ' + str(today['temperatureMax']) + '℃')
    today_set.append('最低気温: ' + str(today['temperatureMin']) + '℃')
    today_set.append('降水確率: ' + str(today['precipProbability'] * 100) + '%')
    today_set.append('湿度: ' + str(today['humidity'] * 100) + '%')
    today_set.append('気圧: ' + str(today['pressure']) + 'hPa')
    ReturnText = []
    ReturnText.append('\n'.join(curr_set))
    ReturnText.append('\n'.join(today_set))

    return ReturnText
