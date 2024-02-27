import ray
ray.init()

from autorecipe.genai.utils import filter_questions_using_reference, filter_and_sort_questions

from datasketch import MinHashLSH, MinHash
num_perm = 128

from sentence_transformers import SentenceTransformer
val_model = SentenceTransformer('all-mpnet-base-v2')

# Example usage
import pandas as pd
df = pd.read_csv('./aircompressor/genai_questions_bank_aircompressor_llama.csv')
df = pd.read_csv('./aircompressor/genai_questions_bank_aircompressor_mixtral.csv')

string_list = list(df['questions'])
print (len(string_list))

'''
lsh = MinHashLSH(threshold=0.6, num_perm=num_perm)

# Create MinHashes for each string and add them to the LSH index
for i, string in enumerate(string_list):
    minhash = MinHash(num_perm=num_perm)
    for word in string.split():
        minhash.update(word.encode('utf-8'))
    lsh.insert(str(i), minhash)

for i, query_string in enumerate(string_list):
    query_minhash = MinHash(num_perm=num_perm)
    for word in query_string.split():
        query_minhash.update(word.encode('utf-8'))
    candidate_matches = lsh.query(query_minhash)
    if len(candidate_matches) > 1:
        print (i, len(candidate_matches))
    for candidate_id in candidate_matches:
        if int(candidate_id) != i:
            print (query_string, string_list[int(candidate_id)])

#ans = val_model.encode(string_list[:100], convert_to_tensor=True)
'''

#print (ans)
#filtered_list = filter_and_sort_questions(string_list)
#print(f"Filtered List: {len(filtered_list)}")

filtered_list = filter_questions_using_reference(string_list[:1000],string_list[1000:])
print(f"Filtered List: {len(filtered_list)}")

ray.shutdown()
