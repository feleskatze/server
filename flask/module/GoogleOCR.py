import sys, os
import base64
import json
from requests import Request, Session
import yaml


RequestURL = 'https://vision.googleapis.com/v1/images:annotate?key='


def GoogleOCR(image_path):
    with open('/home/feleskatze/www/key.yaml') as file:
        API_KEY = yaml.load(file)
    image = open(image_path, 'rb').read()
    str_headers = {'Content-Type': 'application/json'}
    batch_request = {'requests': [{'image': {'content': base64.b64encode(image).decode("utf-8")}, 'features': [{'type': 'TEXT_DETECTION'}]}]}
    obj_session = Session()
    obj_request = Request("POST", RequestURL + API_KEY['GoogleApiKey'], data=json.dumps(batch_request), headers=str_headers)
    obj_prepped = obj_session.prepare_request(obj_request)
    obj_response = obj_session.send(obj_prepped, verify=True, timeout=180)


    if obj_response.status_code == 200:
        try:
            result = json.loads(obj_response.text)
            ReturnText = result['responses'][0]['textAnnotations'][0]['description']
        except:
            ReturnText = str(obj_response.status_code) + ' Errorです\n文字列が検出できないか、何らかの問題が発生しています。'
    else:
        ReturnText = 'Error\n' + obj_response.status_code

    return ReturnText
