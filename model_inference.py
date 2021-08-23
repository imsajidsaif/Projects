import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle
import argparse
import os
import json
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.sparse import coo_matrix, hstack

if not os.path.exists('./output'):
    os.mkdir('./output')

if not os.path.exists('./output/temp'):
    os.mkdir('./output/temp')

def get_predictions(data, loaded_model,loaded_transformer):
    similar_address_dict = {}
    similar_found = set()

    for index in range(len(data)-1):
        if index not in similar_found:
            address_1 = data.loc[index, 'address']
            for j in range(index+1,len(data)):
                if j not in similar_found:
                    address_2 = data.loc[j, 'address']
                    t1=loaded_transformer.transform([address_1])
                    t2=loaded_transformer.transform([address_2])
                    X_test = hstack([t1,t2])
                    prediction = loaded_model.predict(X_test)
                    if prediction[0] == 1:
                        similar_found.add(index)
                        similar_found.add(j)
                        if index not in similar_address_dict.keys():
                            similar_address_dict[index] = [j]
                        else:
                            similar_address_dict[index].append(j)
    
    return similar_address_dict, similar_found


def assign_stop_number(data,similar_address_dict, similar_found):
    data['Model Merged Stop'] = np.nan
    stop_number = 1
    for key, values in similar_address_dict.items():
        data.loc[key,'Model Merged Stop'] = stop_number
        for i in similar_address_dict[key]:
            data.loc[i,'Model Merged Stop'] = stop_number
        stop_number = stop_number + 1

    for i in range(len(data)):
        if i not in similar_found:
            data.loc[i,'Model Merged Stop'] = stop_number
            stop_number = stop_number + 1
    
    data['Model Merged Stop'] = data['Model Merged Stop'].astype('int')

    sorted_df = data.sort_values(by='Model Merged Stop')
    sorted_df.reset_index(inplace=True, drop=True)
    sorted_df = sorted_df.drop(['address'], axis = 1)
    return sorted_df

def write_output(output_filepath,merged_data):
    merged_data.to_csv(output_filepath)
    print('[INFO] Output generated successfully!')


if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('file', help='pass the input excel file for a merging')
#     args = parser.parse_args()
    filepath = '246889360_task_number.xlsx'
    model_path="./svm_model.pkl"
    transformer_path="./file.pkl"
    loaded_model = pickle.load(open(model_path, 'rb'))
    loaded_transformer = pickle.load(open(transformer_path, 'rb'))
    final_df = pd.DataFrame(columns = ['Transport Task', 'Shipment Number', 'Name', 'Address', 'City', 'Post Code', 'Merged Stops', 'Stop Number Without Merge', 'Model Merged Stop'])
    start = time.time()

    #if os.path.exists(filepath):
    if len(filepath)>0:
       # print(filepath.split('/')[2])
        complete_data = pd.read_excel(filepath)
        if 'Unnamed: 0' in complete_data.columns:
            complete_data.drop(['Unnamed: 0'],axis=1,inplace=True)
        complete_data.columns = ['Transport Task', 'Shipment Number', 'Name', 'Address', 'City', 'Post Code', 'Merged Stops', 'Stop Number Without Merge']
        complete_data = complete_data.dropna()
        complete_data['Post Code'] = complete_data['Post Code'].astype('int')
        complete_data['Post Code'] = complete_data['Post Code'].astype('str')
        def remove_x_char_from_data(data_frame):# removing 'x' from dataset
            for i in range(len(data_frame)):
                data_frame.Address[i]=data_frame.Address[i].replace('x','')
                #data_frame.City[i]=data_frame.City[i].replace('x','')
                return data_frame
        complete_data=remove_x_char_from_data(complete_data)
        complete_data['address'] = complete_data.apply(lambda x: ''.join(x['Address'] + ' ' + x['City'] + ' ' + str(x['Post Code'])), axis=1)
        grouped_data = complete_data.groupby('Transport Task')
        gb = grouped_data.groups
        merged_data = {'Transport Task' : '', 'Shipment Number': '', 'Name': '', 'Address': '', 'City': '', 'Post Code': '',
         'Merged Stops': '', 'Stop Number Without Merge': '', 'Model Merged Stop': ''}
        for group_key, values in gb.items():
            g_start = time.time()
            print(group_key)
            data = grouped_data.get_group(group_key)
            data.reset_index(drop=True, inplace=True)
            similar_address_dict, similar_found = get_predictions(data, loaded_model,loaded_transformer)
            merged_data = assign_stop_number(data,similar_address_dict, similar_found)
            final_df = final_df.append(merged_data, ignore_index=True)
            outfilename = 'output_' + str(group_key) + '_merged_results_random_forest.csv'
            output_filepath = './output/temp/' + outfilename
            write_output(output_filepath,merged_data)
            g_end = time.time()
            g_total = g_end - g_start
            print('One group completed in {} seconds!'.format(g_total))
        
    if not final_df.empty:
        final_df.reset_index(drop=True,inplace=True)
        final_df.to_excel('result.xlsx')#./output/rf_results_{}'.format(filepath.split('/')[2]))

    else:
        print('[ERROR] Enter a valid path.......Terminating')
    end = time.time()
    total_time = end - start
    print('Process completed in {} seconds'.format(total_time) )
        
        




