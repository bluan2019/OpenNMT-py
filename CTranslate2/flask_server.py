from server import Server
import sys

from flask import Flask, redirect, url_for, request, make_response
from flask_cors import *
import json
app = Flask(__name__)
def cut_list(data, batch_size):
    return [data[x:x+batch_size] for x in range(0, len(data), batch_size)]

@app.route('/new_trans/en2zh', methods = ['POST', 'GET'])
def predict_en2zh():
    text_list = request.get_json()['text_list']
    result = server_en2zh.predict(text_list)
    result = [e.replace(' ', '') for e in result]
    return json.dumps(result, ensure_ascii=False) 

@app.route('/new_trans/zh2en', methods = ['POST', 'GET'])
def predict_zh2en():
    text_list = request.get_json()['text_list']
    result = server_zh2en.predict(text_list)
    return json.dumps(result, ensure_ascii=False) 

if __name__ == '__main__':
    path_translator = "zh2en_ctranslate2"
    path_bpe="zh.bpe"

    server_zh2en = Server("zh2en_ctranslate2", "zh.bpe")
    server_en2zh = Server("en2zh_ctranslate2", "en.bpe")
    
    port = 9124
    CORS(app, supports_credentials=True)
    app.run(host='0.0.0.0', debug=True, port=port)



