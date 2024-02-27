import pandas as pd
import json

df1 = pd.read_csv('./Generated_result_llama_2_70b_chat.csv')
df2 = pd.read_csv('./Generated_result_granite_13b_chat_v2.csv')
df3 = pd.read_csv('./Generated_result_mixtral_8x7b_instruct_v01_q.csv')
df4 = pd.read_csv('./Generated_result_granite_13b_labrador_rc.csv')
problemcodeLst = list(df1['problemcode'])

names = ['llama', 'granite_v2', 'mitral_v', 'labradore']

def get_prediction(slist, rank=3):
    '''
    '''
    ans = []
    try:
        for item in slist:
            if item[1] <= rank:
                ans.append(item[0])
    except:
        pass
    return ans

ranks = [1, 3, 5, 10]

def get_lists(data):
    extracted_data = []
    # Iterate over the tuples in the list
    for key, value in data:
        # If the value is a list of dictionaries
        if isinstance(value, list):
            # Extract 'failure code' and 'rank' values from each dictionary
            failure_codes = [(d['failure code'], d['rank']) for d in value]
            # Append a tuple with the key and extracted data to the result list
            extracted_data.append((key, failure_codes))
        else:
            # If the value is not a list of dictionaries, simply append the tuple to the result list
            extracted_data.append((key, value))
    return extracted_data

for index, df in enumerate([df1, df2, df3, df4]):
    print (names[index])
    for rk in ranks:
        total_in = 0
        for i in range(df.shape[0]):
            list_of_tuples = []
            json_res = df['0'][i]
            #print (json_res)
            if i > 7:
                continue
            start = json_res.find('{')
            end = json_res.rfind('}')
            try:
                #print('-----------------------')
                #print (json_res[start:end+1])
                json_res = json.loads(json_res[start:end+1])
                list_of_tuples = [(key, value['rank']) if isinstance(value, dict) else (key, value) for key, value in json_res.items()]
                if isinstance(list_of_tuples[0], tuple) and 'failure' in list_of_tuples[0][0]:
                    list_of_tuples = get_lists(list_of_tuples)
                    list_of_tuples = list_of_tuples[0][1]
            except:
                pass
            finally:
                pass   
            aans = []
            print (list_of_tuples)
            aans = get_prediction(list_of_tuples, rank=rk)
            if problemcodeLst[i] in aans:
                total_in = total_in + 1
        print (f"Total in: {total_in}, rans: {rk}")

