
import json
import pandas as pd
import numpy as np
import re
import string
import os
import sys
from pyspark.sql import SparkSession
import pyspark
from pyspark.sql.functions import col
def read_json(file_name): ## helper method to read json files more efficiently 
    df = pd.DataFrame()
    chunk_size = 100000
    for chunk in pd.read_json(file_name, chunksize=chunk_size, lines=True):
        df = pd.concat([df, chunk])
        break
    return df

os.environ['KAGGLE_USERNAME'] = "xxxxxx"
os.environ['KAGGLE_KEY'] = "xxxxxx"
!kaggle datasets download -d 'yelp_academic_dataset_review.json'
!kaggle datasets download -d 'yelp_academic_dataset_user.json'
!kaggle datasets download -d 'yelp_academic_dataset_business.json'


#ENVIRONMENT VARIABLE
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
#GLOBAL VARIABLE
k = 10 
t = 100

#SLOW FUNCTION1 (FOR A FAST TESTING)
def populate_utility_matrix(df):
    # Mapping degli ID user e business in indici
    user_index_map = {user_id: idx for idx, user_id in enumerate(df['user_id'].unique())}
    business_index_map = {business_id: idx for idx, business_id in enumerate(df['business_id'].unique())}
    
    # Creazione di una utility matrix vuota
    utility_matrix = np.full((len(user_index_map), len(business_index_map)), np.nan)
    
    # Riempimento della utility matrix con i punteggi delle recensioni
    for _, row in df.iterrows():
        user_index = user_index_map[row['user_id']]
        business_index = business_index_map[row['business_id']]
        utility_matrix[user_index, business_index] = float(row['stars'])
    
    return utility_matrix,user_index_map,business_index_map
#SLOW FUNCTION2 (FOR A FAST TESTING)
def cosine_similarity(matrix):
    # Calcolo delle norme delle colonne
    normalized_matrix = np.nan_to_num(matrix)
    norms = np.linalg.norm(normalized_matrix, axis=0)
    
    # Normalizzazione delle colonne
    normalized_matrix = normalized_matrix / norms[np.newaxis, :]   
    # Calcolo della cosine similarity
    similarity_matrix = np.dot(normalized_matrix.T, normalized_matrix)
    
    return similarity_matrix
#DATASET FOR MATRIX : 

#UTIL FOR COMPUTING 
def create_dict(values):
    dict = {}
    for key,value in values:
        dict.update({key:value})
    return dict
def normalize_row(values):
    row_values = list(values)
    mean = sum(value[1] for value in row_values) / len(row_values)
    updated_values = [(value[0], value[1] - mean) for value in row_values]
    return updated_values
#USED TO MAP NORMALIZED UTILITY MATRIX IN A COLUMN REPRESENTATION FOR SIMILARITY COMPUTING
def transform_row(row):
    user, business_ratings = row    
    return [(business, (user, rating)) for business, rating in business_ratings]
#########################################
############# MAIN FUNCTION #############
#########################################

#Create Utility Matrix in sparse form
def map_to_indices(row,user_map,business_map):
        user_id = row.user_id
        business_id = row.business_id
        user_index = user_map.get(user_id, -1)  
        business_index = business_map.get(business_id, -1) 
        if user_index == -1 or business_index == -1:
            pass
        else :        
            return (user_index, (business_index, row.stars))

def calculate_similarity(pair,columns_broadcast):
    columns = columns_broadcast.value 
    business1, business2 = pair 
    if business1==business2:
        return None
    dict1 = columns[business1] 
    dict2 = columns[business2]
    common = set(dict1.keys()) & set(dict2.keys())
    if not common:
        return None
    else :         
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        for key in common:            
            numerator += dict1[key] * dict2[key]
            denominator1 += dict1[key] ** 2
            denominator2 += dict2[key] ** 2
        if denominator1 == 0 or denominator2 == 0:
            return None
        else:
            similarity = numerator / (denominator1 ** 0.5 * denominator2 ** 0.5)
            return (business1,(business2,similarity) )

#Create Blank Prediction Matrix
def upgrade_prediction(row,global_index,similarity_rdd,k):
    similarity_matrix = similarity_rdd.value
    result = []
    
    ind = set(global_index) - set(row.keys())
    avg = sum(row.values())/len(row)
    for i in ind:            
            if i in similarity_matrix:                
                similarity_dict = similarity_matrix[i]
                #TOP_K_SIMILARITIES :
                sorted_dict = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)
                top_k = dict(sorted_dict[:k])
                common = set(top_k.keys()) & set(row.keys())
                numerator=0
                denominator=0
                for key in common:
                    numerator += row[key]*top_k[key]
                    denominator += top_k[key]                                 
                score = avg + (numerator / denominator) if denominator != 0 and numerator != 0 else -5
                if score != -5:
                    result.append({i: score})
            else : 
                print('negative result')
            
    
    return sorted(result, key=lambda x: list(x.values())[0], reverse=True)

#PYSPARK CONFIGURATION
conf = pyspark.SparkConf()
conf.setMaster("local[*]")
conf.setAppName("MyApp")
conf.set("spark.driver.memory", "8g") #has to be enough to contain the dataset
spark = SparkSession.builder.config(conf=conf).getOrCreate()

#READ DATAFRAME AND CREATE DICTS FORM USER AND BUSINESS MAP : 
df = spark.read.json("yelp_academic_dataset_review.json")
#PREPROCESS : 
df_u = spark.read.json("yelp_academic_dataset_user.json")
df_b = spark.read.json("yelp_academic_dataset_business.json")

invalid_users  = df_u.filter(df_u['num_reviews'] < t).select('user_id').collect()
invalid_business = df_b.filter(df_b['num_reviews'] < t).select('business_id').collect()

df = df.filter(~col('user_id').isin(invalid_users))
df = df.filter(~col('business_id').isin(invalid_business))


#CREATE MAP FOR ID TO INTEGER
unique_user_ids = df.select('user_id').distinct().rdd.map(lambda row: row[0]).collect()
unique_business_ids = df.select('business_id').distinct().rdd.map(lambda row: row[0]).collect()

user_map = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
business_map = {business_id: idx for idx, business_id in enumerate(unique_business_ids)}

#UTILITY MATRIX BY BUSINESS : 
utility_row= df.rdd.map(lambda row: map_to_indices(row, user_map, business_map)) \
                        .filter(lambda pair: pair[0] != -1 and pair[1][0] != -1)\
                        .groupByKey().mapValues(lambda values : normalize_row(values))

columns = utility_row.flatMap(transform_row).groupByKey().mapValues(lambda values : create_dict(values))

#convert tuple into dict to optimize the operation
utility_row = utility_row.mapValues(lambda values : create_dict(values))

business_combinations = columns.keys() \
    .cartesian(columns.keys()) \
#Broadcast of Utility matrix by column
columns_broadcast = spark.sparkContext.broadcast(columns.collectAsMap())
#similarity matrix : 
similarities = business_combinations.map(lambda pair: calculate_similarity(pair, columns_broadcast)).filter(lambda x: x is not None)
similarities_row = similarities.groupByKey().mapValues(lambda values : create_dict(values))

#Broadcast of similarity matrix for prediction computing
similarity_rdd = spark.sparkContext.broadcast(similarities_row.collectAsMap())
#index for blank value computing
global_index =  business_map.values()

upgraded_matrix = utility_row.map(lambda row: (row[0],upgrade_prediction(row[1],global_index,similarity_rdd,k)))


upgraded_matrix.saveAsTextFile("output")





#
