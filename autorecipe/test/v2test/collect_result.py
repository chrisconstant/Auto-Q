import pandas as pd
import re
df = pd.read_csv('./topk_Generated_result_granite_13b_chat_v2.csv')
df = pd.read_csv('./topk_Generated_result_granite_13b_labrador_rc.csv')
df = pd.read_csv('./topk_Generated_result_llama_2_70b_chat.csv')
df = pd.read_csv('./topk_Generated_result_mixtral_8x7b_instruct_v01_q.csv')
#print (df)

answer = list(df['0.1'])
problecode = list(df['problemcode'])

total_c = 0
for i in range(len(answer)):
    lines_starting_with_numbers = re.findall(r'^\d+\..*$', answer[i], re.MULTILINE)
    ans = []
    for line in lines_starting_with_numbers:
        ans.append(line.strip().split(' ')[1].split(',')[0].split(':')[0])
    print (ans[:5], problecode[i])
    if problecode[i] in ans[:5]:
        total_c = total_c + 1
print (total_c)