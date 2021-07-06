# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:56:32 2021

@author: NicNeo
"""

from flask import Flask, render_template, jsonify, request
from evaluation import get_response

app = Flask(__name__)
