from flask import Flask
from flask_cors import CORS
from flask import request
import json
import os
from util.plot import plot,plot_path
import base64
from src.ged_remote import computation
 
app = Flask(__name__, static_url_path='')
 
@app.route('/hello')
def hello():
    return "hello world"
 
@app.route('/upload',methods=['POST'])
def upload():
    info = request.get_data()
    info_string = info.decode()
    index1_start = info_string.find('<?xml')
    index1_end = info_string.find('</gexf>')
    if (plot(info_string[index1_start:index1_end+len('</gexf>')],'temp1.png')):
        return "Unresolved File!"
    else:
        img_file = open('temp1.png','rb') 
        img_stream = img_file.read()
        img_stream = base64.b64encode(img_stream)
        img_file.close()

    index2_start = info_string.find('<?xml',index1_end+1)
    index2_end = info_string.find('</gexf>',index1_end+1)
    img_stream1 = ''
    if not (index2_start == index2_end):
        if (plot(info_string[index2_start:index2_end+len('</gexf>')],'temp2.png')):
            return "Unresolved File!"
        else:
            img_file1 = open('temp2.png','rb') 
            img_stream1 = img_file1.read()
            img_stream1 = base64.b64encode(img_stream1)
            img_file1.close()
    else: 
        return img_stream
    img_stream = img_stream + bytes(','.encode()) + img_stream1
    return img_stream

@app.route('/query',methods=['POST'])
def query():
    data = json.loads(request.get_data(as_text=True))
    if 'function' not in data.keys():
        function = None
    else:
        function = data['function']
    if 'beamsize' not in data.keys():
        beamsize = None
    else:
        beamsize = data['beamsize']
    path, cost = computation(data['alg'],data['method'],function,beamsize)
    plot_path(path)
    img_file = open('test.png','rb') 
    img_stream = img_file.read()
    img_stream = base64.b64encode(img_stream)
    img_file.close()
    img_stream = img_stream + bytes(','.encode()) + bytes(str(cost).encode())
    return img_stream

if __name__ == '__main__':
    CORS(app, supports_credentials=True)
    app.run(host='0.0.0.0', port=5000)
