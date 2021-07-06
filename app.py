# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:56:32 2021

@author: NicNeo
"""

from flask import Flask, render_template, jsonify, request
from evaluation import get_response, bot_name
from waitress import serve

app = Flask(__name__)

app.config.from_pyfile('config.py')

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html', **locals())

@app.route('/chat', methods=["POST"])
def chatbotResponse():
    if request.method == 'POST':
        question = request.form['question']
        response = get_response(question)
    return jsonify({"response": response, "bot_name": bot_name})

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)
    #app.run(host='0.0.0.0', port=8080, debug=True)