import sys
import os
import pickle
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import mysql.connector

# Load environment variables
load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info('Reading SQL database started')
    try:
        mydb = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=db
        )
        logging.info(f"Connection Established: {mydb}")
        df = pd.read_sql_query('SELECT * FROM students', mydb)
        print(df.head())
        return df
    except Exception as ex:
        raise CustomException(ex, sys)

def save_object(file_path, obj):
    """Save a Python object to disk using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")
    except Exception as ex:
        raise CustomException(ex, sys)

def load_object(file_path):
    """Load a Python object from disk using pickle."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as ex:
        raise CustomException(ex, sys)
