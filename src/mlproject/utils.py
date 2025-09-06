import sys
import os
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import mysql.connector

load_dotenv()
host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv('db')


def read_sql_data():
    logging.info('reading SQL database is started')
    try:
        mydb=mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("connection Established: {mydb}")
        df=pd.read_sql_query('Select * from students',mydb)
        print(df.head())
        
        return df
    except Exception as ex:
            raise CustomException(ex,sys)

