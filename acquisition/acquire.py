import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password


############################################################################

# 

############################################################################


def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'    
    
    
    
    # Re-acquire Database        
        
def new_data(sql_query, db):
    '''
    This function takes in a SQL query and a database and returns a dataframe.
    
    Parameters:
    --------
    sql_query: DOCSTRING
        SQL query that collects the desired dataset to be acquired
    db: str
        The name of the target database
        
    '''
    return pd.read_sql(sql_query, get_connection(db))
        
        
        
def get_data(cached=False):
    '''
    This function reads in data from Codeup database and writes data to a csv file if cached == False or if cached == True reads in titanic df from a csv file, returns df.
    '''
    if cached == False or os.path.isfile(db + '.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_data(sql_query, db)
        
        # Write DataFrame to a csv file.
        df.to_csv(db + '.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv(db + '.csv', index_col=0)
        
    return df