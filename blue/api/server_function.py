
def date_to_list(x2):
    lenght=len(x2)
    print("my lenght is",lenght)
    list_date=[]
    for x in x2:
        list_date.append(str(x))
    return list_date

from flask import Flask
from flask_apscheduler import APScheduler
import datetime

app = Flask(__name__)

#function executed by scheduled job
def my_job():
    Function_for_file_generation(gpath)

if (__name__ == "__main__"):
    scheduler = APScheduler()
    scheduler.add_job(func=my_job, args=['job run'], trigger='interval', id='job', seconds=10)
    scheduler.start()
    app.run(port = 8000)





