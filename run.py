from blue import app
import os
import sys

dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
path = dirname + "/api"
sys.path.append(path)

from blue.api.For_Files_Gen import Function_for_file_generation


import threading
def Repeat_Insertion():
    threading.Timer(3600,Repeat_Insertion).start()
    Function_for_file_generation()


if __name__ == "__main__":
    #Repeat_Insertion()
    app.run(debug = True,use_reloader=False)
