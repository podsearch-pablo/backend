import os
from flask_cors import CORS

from flask import Flask
from flask import request
import flask

import json
import sys
sys.path.insert(1, './flaskr')
from main import masterSearch
import modules



def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    CORS(app, support_credentials=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass


    # a simple page that says hello
    @app.route('/hello')
    def hello():
        print("CALLED")
        return 'Hello, World!'


    # perform a search for passed in search term
    @app.post('/details')
    def details():
        print("plaintext search")
        data = request.json
        val = json.dumps(masterSearch(data['searchterm']))
        return val

    
    # ask Colin and Samir AI a context-specific question, recieve an answer + relevant clip
    @app.post('/query')
    def query():
        print("querying")
        data = request.json

        return modules.answer_question(data['query'])
    
    # same as query but answer is in the form of a poem
    @app.post('/poem')
    def querypoem():
        print("querying for a poem")
        data = request.json

        return modules.answer_question_poem(data['query'])


        





    return app