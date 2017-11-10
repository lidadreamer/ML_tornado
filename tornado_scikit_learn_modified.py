#!/usr/bin/python
'''Starts and runs the scikit learn server'''

# For this to run properly, MongoDB must be running
#    Navigate to where mongo db is installed and run
#    something like $./mongod --dbpath "../data/db"
#    might need to use sudo (yikes!)

# database imports
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


# tornado imports
import tornado.web
from tornado import gen
from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from concurrent.futures import ThreadPoolExecutor
from tornado.concurrent import run_on_executor

# custom imports
from basehandler import BaseHandler
import sklearnhandlers as skh
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from bson.binary import Binary
import json
import numpy as np

# Setup information for tornado class
define("port", default=8000, help="run on the given port", type=int)

thread_pool = ThreadPoolExecutor()
# Utility to be used when creating the Tornado server
# Contains the handlers and the database connection
class Application(tornado.web.Application):
    def __init__(self):
        '''Store necessary handlers,
           connect to database
        '''

        handlers = [(r"/[/]?", BaseHandler),
                    (r"/Handlers[/]?",        skh.PrintHandlers),
                    (r"/AddDataPoint[/]?",    skh.UploadLabeledDatapointHandler),
                    (r"/GetNewDatasetId[/]?", skh.RequestNewDatasetId),
                    (r"/UpdateModel[/]?",     UpdateModelForDatasetId),     
                    (r"/PredictOne[/]?",      skh.PredictOneFromDatasetId),               
                    ]

        self.handlers_string = str(handlers)

        try:
            self.client  = MongoClient(serverSelectionTimeoutMS=50) # local host, default port
            print(self.client.server_info()) # force pymongo to look for possible running servers, error if none running
            # if we get here, at least one instance of pymongo is running
            self.db = self.client.sklearndatabase # database with labeledinstances, models
            
        except ServerSelectionTimeoutError as inst:
            print('Could not initialize database connection, stopping execution')
            print('Are you running a valid local-hosted instance of mongodb?')
            #raise inst
        
        self.clf = {} # the classifier model (in-class assignment, you might need to change this line!)
        # but depending on your implementation, you may not need to change it  ¯\_(ツ)_/¯

        settings = {'debug':True}
        tornado.web.Application.__init__(self, handlers, **settings)

    def __exit__(self):
        self.client.close() # just in case

class UpdateModelForDatasetId(BaseHandler):

    _thread_pool = thread_pool

    @gen.coroutine
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''


        acc = yield self.trainModel()
        

        

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":acc})
        self.finish()

    @run_on_executor(executor="_thread_pool")
    def trainModel(self):
        dsid = self.get_int_arg("dsid",default=0)
        classifier_type = self.get_int_arg("classifier",default=1)

        # create feature vectors from database
        f=[];
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            f.append([float(val) for val in a['feature']])

        # create label vector from database
        l=[];
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            l.append(a['label'])

        # fit the model to the data
        print(classifier_type)
        if classifier_type == 0: 
            c1 = KNeighborsClassifier(n_neighbors=3)
            print('Training KNN Classifier')
        
        elif classifier_type == 1: 

            c1 = SVC()
            print('Training SVM Classifier')
        
        
        acc = -1;
        if l:
            c1.fit(f,l) # training
            lstar = c1.predict(f)
            self.clf[dsid] = c1
            acc = sum(lstar==l)/float(len(l))
            bytes = pickle.dumps(c1)
            self.db.models.update({"dsid":dsid},
                {  "$set": {"model":Binary(bytes)}  },
                upsert=True)
        return acc



def main():
    '''Create server, begin IOLoop 
    '''
    tornado.options.parse_command_line()
    http_server = HTTPServer(Application(), xheaders=True)
    http_server.listen(options.port)
    IOLoop.instance().start()

if __name__ == "__main__":
    main()
