import mysql.connector
import pandas as pd
from utils import lin_log_interp
import numpy as np
import ast

class DBQuerier(object):

    def __init__(self,database,assetId,host='10.0.0.17',debug=True):

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
        self.host = host

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

        # print(query)

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

    def select_fft_features(self,
                            minDate=None,
                            stdev=False,
                            labeled=True,
                            sensorId=None,
                            limit=None,
                            descending_order=True,
                            fft_interval=None,
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

        if limit is not None:
            limit_str = "limit {}".format(limit)
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
        VibrationState.values as vibState
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
        """ + minDate_str + """ """ + interval_str + """
        order by """ + table + """.dateTime """ + desc_str + " " + limit_str + """; """
        
        print(query)

        cursor = self.execute_query(query)
        data=cursor.fetchall()

        if len(data) <= 0:
            return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
         
 
        sensorId = np.array([[data[i][0] for i in range(len(data))]]).T
        dateTime = np.array([[data[i][1] for i in range(len(data))]]).T
        fftVals = np.array([ast.literal_eval(data[i][2]) for i in range(len(data))])
        rmsVals = np.array([[data[i][3] for i in range(len(data))]]).T
        vibState = np.array([[data[i][4] for i in range(len(data))]]).T

        columns = ['FFT-{}'.format(i) for i in range(fftVals.shape[1])]
        columns = ['dateTime'] + columns + ['RMS','sensorId','VibState']
        
        fftVals = lin_log_interp(fftVals)

#         self.minQueriedDate = np.amin(dateTime).strftime('%Y-%m-%d %H:%M:%S.%f')

        statsFeatures = np.hstack((dateTime,fftVals,rmsVals,sensorId,vibState))

        featuresDF = pd.DataFrame(data=statsFeatures,columns=columns)
        featuresDF = featuresDF.set_index('dateTime')
        featuresDF.index = pd.to_datetime(featuresDF.index)
    
        cursor.close()
        self.disconnect()

        sensorIdDF = featuresDF.loc[:, featuresDF.columns == 'sensorId']
        vibStateDF = featuresDF.loc[:, featuresDF.columns == 'VibState']
        featuresDF = featuresDF.drop(['RMS','sensorId','VibState'],axis=1).astype(float)
        
        return featuresDF,sensorIdDF,vibStateDF

    def select_ml_features(self,labeled=True,sensorId=None,limit=None):

        if self.debug:
            print(self.minQueriedDate)

        if labeled == True:
            vibStateSelect = """exists 
            (SELECT VibrationState.values from db18.VibrationState WHERE VibrationState.dateTime = FFT.dateTime)"""
        else:
            vibStateSelect = """FFT.dateTime < '""" + self.minQueriedDate + """'
            and not exists 
            (SELECT VibrationState.values from db18.VibrationState WHERE VibrationState.dateTime = FFT.dateTime)"""

        if sensorId is not None:
            sensorId_str = "and db18.VibrationSkewness.sensorId = '" + sensorId + "'"
        else: 
            sensorId_str = ''

        if limit is not None:
            limit_str = "limit {}".format(limit)
        else:
            limit_str = ""

        query = """SELECT FFT.sensorId, 
        FFT.dateTime AS dateTime, 
        FFT.values AS fftVals, 
        RMS.values AS rmsVals, 
        VibrationMean.values as vibMeanVals, 
        VibrationSkewness.values as vibSkewVals, 
        VibrationKurtosis.values as vibKurtVals, 
        VibrationVariance.values as vibVarVals, 
        VibrationState.values as vibState
        FROM db18.VibrationSkewness
        INNER JOIN db18.FFT ON
        db18.FFT.dateTime = db18.VibrationSkewness.dateTime 
        and db18.VibrationSkewness.assetId = '""" + self.assetId + """' 
        """ + sensorId_str + """
        INNER JOIN db18.RMS ON 
        db18.RMS.dateTime = db18.VibrationSkewness.dateTime 
        INNER JOIN db18.VibrationMean ON 
        db18.VibrationMean.dateTime = db18.VibrationSkewness.dateTime 
        INNER JOIN db18.VibrationKurtosis ON 
        db18.VibrationKurtosis.dateTime = db18.VibrationSkewness.dateTime 
        INNER JOIN db18.VibrationVariance ON 
        db18.VibrationVariance.dateTime = db18.VibrationSkewness.dateTime 
        LEFT JOIN db18.VibrationState ON 
        db18.VibrationState.dateTime = db18.VibrationSkewness.dateTime 
        where """ + vibStateSelect + """
        """ + self.debug_str + """
        order by FFT.dateTime desc """ + limit_str + """; """

        cursor = self.execute_query(query)
        data=cursor.fetchall()

        if len(data) <= 0:
            return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
         
        columns = ['FFT-{}'.format(i) for i in range(257)]
        columns = ['dateTime'] + columns + ['RMS', 'Mean', 'Skew', 'Kurtosis', 'Variance','sensorId','VibState']
        
        sensorId = np.array([[data[i][0] for i in range(len(data))]]).T
        dateTime = np.array([[data[i][1] for i in range(len(data))]]).T
        fftVals = np.array([ast.literal_eval(data[i][2]) for i in range(len(data))])
        rmsVals = np.array([[data[i][3] for i in range(len(data))]]).T
        vibMeanVals = np.array([[data[i][4] for i in range(len(data))]]).T
        vibSkewVals = np.array([[data[i][5] for i in range(len(data))]]).T
        vibKurtVals = np.array([[data[i][6] for i in range(len(data))]]).T
        vibVarVals = np.array([[data[i][7] for i in range(len(data))]]).T
        vibState = np.array([[data[i][8] for i in range(len(data))]]).T

        vibKurtVals = np.abs(vibKurtVals)
        vibSkewVals = np.abs(vibSkewVals)
                
        fftVals = lin_log_interp(fftVals)

        self.minQueriedDate = np.amin(dateTime).strftime('%Y-%m-%d %H:%M:%S.%f')

        statsFeatures = np.hstack((dateTime,fftVals,rmsVals,vibMeanVals,vibSkewVals,vibKurtVals,vibVarVals,sensorId,vibState))

        featuresDF = pd.DataFrame(data=statsFeatures,columns=columns)
        featuresDF = featuresDF.set_index('dateTime')
        featuresDF.index = pd.to_datetime(featuresDF.index)
    
        cursor.close()
        self.disconnect()

        sensorIdDF = featuresDF.loc[:, featuresDF.columns == 'sensorId']
        vibStateDF = featuresDF.loc[:, featuresDF.columns == 'VibState']
        featuresDF = featuresDF.drop(['sensorId','VibState'],axis=1).astype(float)
        
        return featuresDF,sensorIdDF,vibStateDF

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
