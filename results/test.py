import ray
ray.init()

def is_string_included(substring, fullstring):
    return substring in fullstring

@ray.remote
def check_included(string, string_list):
    for other_string in string_list:
        if string != other_string and is_string_included(string, other_string):
            return True
    return False

def remove_included_substrings(string_list):
    # Parallelize the check_included function
    string_list_sorted = sorted(string_list, key=len, reverse=True)

    num_strings = len(string_list)
    futures = [check_included.remote(string, string_list) for string in string_list_sorted[:int(num_strings * 0.9)]]
    results = ray.get(futures)
    
    # Collect strings that are not included in any other string
    included_indices = set()  # Indices of strings that are included in other strings
    for i, is_included in enumerate(results):
        if is_included:
            included_indices.add(i)

    # Remove strings that are included in other strings
    result = [string for i, string in enumerate(string_list) if i not in included_indices]
    return result

# Example usage
import pandas as pd
df = pd.read_csv('./aircompressor/genai_questions_bank_aircompressor_llama.csv')
string_list = list(df['questions'])
print (len(string_list))

filtered_list = remove_included_substrings(string_list)
print(f"Filtered List: {len(filtered_list)}")

ray.shutdown()
