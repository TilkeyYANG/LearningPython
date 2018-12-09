
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 22:36:20 2018

@author: Invisible-Tilkey
"""


from pyspark import SparkContext
from pyspark import SparkConf
sc = SparkContext.getOrCreate()

import os 
os.chdir('D:\ML\ALGOS\PySpark\datas')
cwd = os.getcwd()

from pyspark.sql import SparkSession 

spark = SparkSession.builder.appName("dataFrameApply").getOrCreate() #设置文件路径
flightPerfFilePath = cwd + "/departuredelays.csv"
airportsFilePath =cwd + "/airport-codes-na.txt"

#获得机场数据集
airports = spark.read.csv(airportsFilePath,header='true',inferSchema='true',sep='\t') 
airports.registerTempTable("airports") #获得起飞延时数据集
flightPerf = spark.read.csv(flightPerfFilePath,header='true') 
flightPerf.registerTempTable("FlightPerformance") #缓存起飞延迟数据集
flightPerf.cache()

#通过城市和起飞代码查询航班延误的总数
#（华盛顿州）
spark.sql("""select a.city, f.origin
    from FlightPerformance f 
    join airports a 
    on a.IATA = f.origin """).show()
