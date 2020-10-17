from flask import Blueprint,jsonify, request
from flask_restful import Api,Resource
import pandas as pd
from werkzeug.utils import secure_filename
#global gpath

import sys 
import os
import json


dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
print(dirname)
path = dirname + "/api"
#print(path)
sys.path.append(path)
#gpath=path

from plot_on_server import main_function

mod = Blueprint('api',__name__)
api = Api(mod)
df = None



class Plt(Resource):
    def post(self):
        postedData=request.get_json()
        instrument=postedData["Instrument"]
        print(instrument)
        retJson = main_function(instrument)
        print("retJson",retJson)
        return retJson

api.add_resource(Plt, "/plot_one")
