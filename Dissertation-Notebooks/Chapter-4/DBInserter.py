import mysql.connector
import pandas as pd
import numpy as np
import ast

class DBInserter(object):

    def __init__(self,database,debug=True):

        if debug:
            self.debug_str = ""
        else:
            self.debug_str = "and VibrationState.PCAFit IS NULL"

        self.debug = debug

        self.database = database
        self.reset_min_date()

        self.user = 'dnewman'
        self.password = 'Convolve7691!'
        self.host = '127.0.0.1'

    def connect(self):
        self.mydb = mysql.connector.connect(user=self.user,password=self.password,
                                host=self.host,database=self.database)

    def disconnect(self):
        self.mydb.close()

    def reset_min_date(self):
        self.minQueriedDate = "2999-01-01 00:00:00.000000"

    def insert_labels(self,dates,labels):

        insert_vals = ["('" + dates[i] + "', '" + self.assetId + "', '" + labels[i] + "')," for i in range(len(dates))]
        # insert_vals = insert_vals[0]
        insert_vals = ''.join(insert_vals)[:-1]
        
        
        query = """INSERT INTO """ + self.database + """.VibrationState
                (`dateTime`,`assetId`,`values`) VALUES 
                """ + insert_vals + """;"""

        self.execute_query(query)

    def select_labels(self):
        query = """SELECT VibrationState.values, VibrationState.dateTime, RMS.values as rmsVals
                FROM """ + self.database + """.VibrationState
                INNER JOIN """ + self.database + """.RMS ON
                db18.RMS.dateTime = db18.VibrationState.dateTime 
                where VibrationState.assetId = '""" + self.assetId + """'
                order by dateTime desc;"""
        
        cursor = self.execute_query(query)
        data=cursor.fetchall()
        columns = ['dateTime','RMS','VibState']
        vibState = np.array([[data[i][0] for i in range(len(data))]]).T
        dateTime = np.array([[data[i][1] for i in range(len(data))]]).T
        rmsVals = np.array([[data[i][2] for i in range(len(data))]]).T

        Data = np.hstack((dateTime,rmsVals,vibState))

        resultDF = pd.DataFrame(data=Data,columns=columns)
        # resultDF['dateTime'] = pd.to_datetime(resultDF['dateTime'])

        return resultDF

    def insert_fft(self,dateTime,assetId,sensorId,values,frequencyInterval):
        query = """INSERT INTO """ + self.database + """.FFT (`frequencyInterval`,`dateTime`,`assetId`,`values`,`sensorId`) 
                VALUES (""" + str(frequencyInterval) + """,'""" + dateTime + """', 
                '""" + assetId + """', '""" + values + """', '""" + sensorId + """');"""  
                
        self.execute_query(query)
        
    def insert_fft_std(self,dateTime,assetId,sensorId,values,frequencyInterval):
        query = """INSERT INTO """ + self.database + """.FFTSTD (`frequencyInterval`,`dateTime`,`assetId`,`values`,`sensorId`) 
                VALUES (""" + str(frequencyInterval) + """,'""" + dateTime + """', 
                '""" + assetId + """', '""" + values + """', '""" + sensorId + """');"""  
           
        self.execute_query(query)
        
    def insert_vibration(self,dateTime,assetId,sensorId,values,samplingInterval):
        query = """INSERT INTO """ + self.database + """.Vibration (`samplingInterval`,`dateTime`,`assetId`,`values`,`sensorId`) 
                VALUES (""" + str(samplingInterval) + """,'""" + dateTime + """', 
                '""" + assetId + """', '""" + values + """', '""" + sensorId + """');"""  
           
        self.execute_query(query)
        
    def insert_value(self,dbTable,dateTime,assetId,sensorId,values):
        query = """INSERT INTO """ + self.database + """."""+ dbTable + """ (`dateTime`,`assetId`,`values`,`sensorId`) VALUES
                ('""" + dateTime + """', '""" + assetId + """', '""" + str(values) + """', '""" + sensorId + """');"""  
        
        self.execute_query(query)
        
    def insert_vib_state(self,dateTime,assetId,values,programName):
        query = """INSERT INTO """ + self.database + """.VibrationState (`dateTime`,`assetId`,`values`,`programName`) VALUES
                ('""" + dateTime + """', '""" + assetId + """', '""" + str(values) + """', '""" + programName + """');"""  
        
        self.execute_query(query)

    def execute_query(self,query):

        self.connect()
        cursor = self.mydb.cursor()
        cursor.execute(query)

        if query.lower().find('insert') != -1:
            self.mydb.commit()
            cursor.close()
            self.disconnect()
            return True
        else:
            return cursor
