import mysql.connector
import pandas as pd
from utils import lin_log_interp
import numpy as np
import ast

class DBQuerier(object):

    def __init__(self,database,assetId,debug=True,limit=None,):

        if debug:
            self.debug_str = ""
        else:
            self.debug_str = "and VibrationState.PCAFit IS NULL"

        self.debug = debug

        self.database = database
        self.assetId = assetId
        self.reset_min_date()

        self.user = 'dnewman'
        self.password = 'Convolve7691!'
        self.host = '10.0.0.17'
        self.limit = limit
        
    def connect(self):
        self.mydb = mysql.connector.connect(user=self.user,password=self.password,
                                host=self.host,database=self.database,use_pure=True)

    def disconnect(self):
        self.mydb.close()

    def reset_min_date(self):
        self.minQueriedDate = "2999-01-01 00:00:00.000000"

    def insert_labels(self,dates,labels):
        
        insert_vals = ["('" + dates[i] + "', '" + self.assetId + "', '" + labels[i] + "')," for i in range(len(dates))]
        insert_vals = ''.join(insert_vals)[:-1]
        
        query = """INSERT INTO """ + self.database + """.VibrationState
                (`dateTime`,`assetId`,`values`) VALUES 
                """ + insert_vals + """;"""
        
        if self.debug:
            print(query)
        else:
            self.execute_query(query)
            
    def insert_labels_program(self,dates,labels, programName):
        
        insert_vals = ["('" + dates[i] + "', '" + self.assetId + "', '" + labels[i] + "', '" + programName + "')," for i in range(len(dates))]
        insert_vals = ''.join(insert_vals)[:-1]
        
        query = """INSERT INTO """ + self.database + """.VibrationState
                (`dateTime`,`assetId`,`values`,`programName`) VALUES 
                """ + insert_vals + """;"""
        
        if self.debug:
            print(query)
        else:
            self.execute_query(query)
            
    def insert_labels_experiment(self,table,dates,
                                      experimentNumber,
                                      experimentName,
                                      toolStatus,
                                      toolSize,
                                      material,
                                      depthOfCut,
                                      surfaceSpeed):
        
        insert_vals = ["('" + dates[i] + "', '" + self.assetId + "', " + experimentNumber + ", '" + experimentName +"', '" + toolStatus + "', '" +  toolSize + "', '" + material + "', '" + depthofCut + "', '" + surfaceSpeed + "')," for i in range(len(dates))]
        
        insert_vals = ''.join(insert_vals)[:-1]
        
        query = """INSERT INTO """ + self.database + """.""" + table + """
                (`dateTime`,`assetId`,`experimentSample`, `experimentName`, `toolStatus`,
                 `toolSize`,`depthOfCut`,`surfaceSpeed`) VALUES 
                """ + insert_vals + """;"""
        
        if self.debug:
            print(query)
        else:
            self.execute_query(query)

    def select_labels(self):
        query = """SELECT VibrationState.values, VibrationState.dateTime, RMS.values as rmsVals
                FROM """ + self.database + """.VibrationState
                INNER JOIN """ + self.database + """.RMS ON
                """ + self.database + """.RMS.dateTime = """ + self.database + """.VibrationState.dateTime 
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

    def select_unique_sensorId(self):

        query = """SELECT DISTINCT(sensorId) FROM """ + self.database + """.RMS
                where assetId = '""" + self.assetId + """';"""

        cursor = self.execute_query(query)
        data=cursor.fetchall()

        sensorId = [data[i][0] for i in range(len(data))]

        return sensorId
    
    def select_frequency_interval(self):
        query = """SELECT frequencyInterval FROM """ + self.database + """.FFT
                where assetId = '""" + self.assetId + """'
                order by id desc limit 1;"""

        cursor = self.execute_query(query)
        data=cursor.fetchall()

        frequencyInterval = float(data[0][0])

        return frequencyInterval
        
    def select_fft_features(self,
                            minDate=None,
                            stdev=False,
                            labeled=True,
                            sensorId=None,
                            limit=None,
                            descending_order=True,
                            fft_interval=None,
                            extra_condition = '',
                            ):
        if self.debug:
            print(self.minQueriedDate)
            
        if fft_interval is not None:
            interval_str = "and frequencyInterval" + fft_interval
        else:
            interval_str = ""
            
        if stdev == True:
            table = 'FFTSTD'
        else:
            table = 'FFT'

        if labeled == True:
            vibStateSelect = """exists 
            (SELECT VibrationState.values from """ + self.database + """.VibrationState WHERE VibrationState.dateTime = """ + table + """.dateTime)"""
        else:
            vibStateSelect = table + """.dateTime < '""" + self.minQueriedDate + """'
            and not exists 
            (SELECT VibrationState.values from """ + self.database + """.VibrationState WHERE VibrationState.dateTime = """ + table + """.dateTime)"""

        if sensorId is not None:
            sensorId_str = "and " + self.database + ".RMS.sensorId = '" + sensorId + "'"
        else: 
            sensorId_str = ''
            
        if minDate is not None:
            minDate_str = "and " + table + ".dateTime > " + minDate
        else:
            minDate_str = ""

        if self.limit is not None:
            limit_str = "limit {}".format(self.limit)
        else:
            limit_str = ""
        
        if descending_order == True:
            desc_str = "desc"
        else:
            desc_str = "asc"

        query = """SELECT """ + table + """.sensorId, 
        """ + table + """.dateTime AS dateTime, 
        """ + table + """.values AS fftVals, 
        RMS.values AS rmsVals,
        VibrationState.values as vibState,
        VibrationState.programName as programName
        FROM """ + self.database + """.""" + table + """
        INNER JOIN """ + self.database + """.RMS ON 
        """ + self.database + """.RMS.dateTime = """ + self.database + """.""" + table + """.dateTime 
        and """ + self.database + """.RMS.assetId = '""" + self.assetId + """' 
        """ + sensorId_str + """
        LEFT JOIN """ + self.database + """.VibrationState ON 
        """ + self.database + """.VibrationState.dateTime = """ + self.database + """.""" + table + """.dateTime 
        where """ + vibStateSelect + """
        """ + self.debug_str + """ 
        and """ + table + """.assetId = '""" + self.assetId + """'
        """ + minDate_str + """ """ + interval_str + """ """ + extra_condition + """
        order by """ + table + """.dateTime """ + desc_str + " " + limit_str + """; """
        
        print(query)

        cursor = self.execute_query(query)
        data=cursor.fetchall()

        if len(data) <= 0:
            return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        
        fftVals = np.array([[]])
         
 
        sensorId = np.array([[data[i][0] for i in range(len(data))]]).T
        dateTime = np.array([[data[i][1] for i in range(len(data))]]).T
        fftVals = np.array([np.array(ast.literal_eval(data[i][2])) for i in range(len(data))])
        rmsVals = np.array([[data[i][3] for i in range(len(data))]]).T.astype(float)
        vibState = np.array([[data[i][4] for i in range(len(data))]]).T
        programName = np.array([[data[i][5] for i in range(len(data))]]).T

        columns = ['FFT-{}'.format(i) for i in range(fftVals.shape[1])]
        columns = ['dateTime'] + columns + ['RMS','sensorId','VibState','programName']

        
        fftVals = lin_log_interp(fftVals).astype(float)

        statsFeatures = np.hstack((dateTime,fftVals,rmsVals,sensorId,vibState,programName))

        featuresDF = pd.DataFrame(data=statsFeatures,columns=columns)
        featuresDF = featuresDF.set_index('dateTime')
        featuresDF.index = pd.to_datetime(featuresDF.index)
    
        cursor.close()
        self.disconnect()

        sensorIdDF = featuresDF.loc[:, featuresDF.columns == 'sensorId']
        vibStateDF = featuresDF.loc[:, featuresDF.columns == 'VibState']
        progNameDF = featuresDF.loc[:, featuresDF.columns == 'programName']
        featuresDF = featuresDF.drop(['RMS','sensorId','VibState'],axis=1)
        
        self.minQueriedDate = np.amin(dateTime).strftime('%Y-%m-%d %H:%M:%S.%f')

        
        return featuresDF,sensorIdDF,vibStateDF, programName
    
    def select_features(self,
                            table,
                            minDate=None,
                            labeled=True,
                            sensorId=None,
                            limit=None,
                            descending_order=True,
                            sampling_interval=None,
                            extra_condition = '',
                            itemInstanceIdStr=''
                            ):
        if self.debug:
            print(self.minQueriedDate)
            
        print(table.lower())
            
        if sampling_interval is not None:
            if table.lower() == 'vibration':
                interval_str = "and samplingInterval" + sampling_interval
            elif table.lower() == 'fft':
                interval_str = "and frequencyInterval" + sampling_interval
        else:
            interval_str = ""
            
        if labeled == True:
            vibStateSelect = """exists 
            (SELECT VibrationState.values from """ + self.database + """.VibrationState WHERE VibrationState.dateTime = """ + table + """.dateTime)"""
        else:
            vibStateSelect = table + """.dateTime < '""" + self.minQueriedDate + """'
            and not exists 
            (SELECT VibrationState.values from """ + self.database + """.VibrationState WHERE VibrationState.dateTime = """ + table + """.dateTime)"""

        if sensorId is not None:
            sensorId_str = "and " + self.database + "." + table + ".sensorId = '" + sensorId + "'"
        else: 
            sensorId_str = ''
            
        if minDate is not None:
            minDate_str = "and " + table + ".dateTime > " + minDate
        else:
            minDate_str = ""

        if self.limit is not None:
            limit_str = "limit {}".format(self.limit)
        else:
            limit_str = ""
        
        if descending_order == True:
            desc_str = "desc"
        else:
            desc_str = "asc"

        query = """SELECT """ + table + """.sensorId, 
        """ + table + """.dateTime AS dateTime, 
        """ + table + """.values AS vals, 
        VibrationState.values as vibState,
        VibrationState.programName as programName""" + itemInstanceIdStr + """
        FROM """ + self.database + """.""" + table + """
        LEFT JOIN """ + self.database + """.VibrationState ON 
        """ + self.database + """.VibrationState.dateTime = """ + self.database + """.""" + table + """.dateTime 
        where """ + vibStateSelect + """
        """ + self.debug_str + """ 
        and """ + table + """.assetId = '""" + self.assetId + """'
        """ + minDate_str + """ """ + interval_str + """ """ + extra_condition + """
        order by """ + table + """.dateTime """ + desc_str + " " + limit_str + """; """
        
        print(query)

        cursor = self.execute_query(query)
        data=cursor.fetchall()

        if len(data) <= 0:
            return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
 
        sensorId = np.array([[data[i][0] for i in range(len(data))]]).T
        dateTime = np.array([[data[i][1] for i in range(len(data))]]).T
        values = np.array([np.array(ast.literal_eval(data[i][2])) for i in range(len(data))])
        vibState = np.array([[float(data[i][3].replace('RPM','')) / 10000. for i in range(len(data))]]).T
        programName = np.array([[data[i][4] for i in range(len(data))]]).T
        
        columns = ['Values-{}'.format(i) for i in range(values.shape[1])]
        
        if table.lower() == 'fft':
            values = lin_log_interp(values).astype(float)
        
        if itemInstanceIdStr != '':
            columns = ['dateTime'] + columns + ['sensorId','VibState','programName','itemInstanceId']
            itemInstanceId = np.array([[data[i][5].replace('Sample-','').split('-')[0] for i in range(len(data))]]).T
            
            u, forward_indices = np.unique(itemInstanceId, return_index=True)
            u, backward_indices = np.unique(itemInstanceId[::-1], return_index=True)
            mask = np.hstack((forward_indices.flatten(),backward_indices.flatten()))
            
            indices = np.arange(0,itemInstanceId.shape[0])
            
            statsFeatures = np.hstack((dateTime,values,sensorId,vibState,programName,itemInstanceId))[indices[np.in1d(indices,mask,invert=True)],:]
        else:
            columns = ['dateTime'] + columns + ['sensorId','VibState','programName']
            statsFeatures = np.hstack((dateTime,values,sensorId,vibState,programName))
            
        featuresDF = pd.DataFrame(data=statsFeatures,columns=columns)
        featuresDF = featuresDF.set_index('dateTime')
        featuresDF.index = pd.to_datetime(featuresDF.index)
    
        cursor.close()
        self.disconnect()

        sensorIdDF = featuresDF.loc[:, featuresDF.columns == 'sensorId']
        vibStateDF = featuresDF.loc[:, featuresDF.columns == 'VibState']
        progNameDF = featuresDF.loc[:, featuresDF.columns == 'programName']
        featuresDF = featuresDF.drop(['sensorId','programName'],axis=1)
        
        self.minQueriedDate = np.amin(dateTime).strftime('%Y-%m-%d %H:%M:%S.%f')

        return featuresDF,sensorIdDF,programName

    def execute_query(self,query):

        self.connect()
        cursor = self.mydb.cursor()
        cursor.execute(query)

        if query.lower().find('insert') != -1:
            self.mydb.commit()
            cursor.close()
            self.disconnect()
            return True
        elif query.lower().find('select') != -1:
            return cursor
        else:
            self.mydb.commit()
            cursor.close()
            self.disconnect()
            return True
