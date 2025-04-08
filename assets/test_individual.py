import pickle

with open('debug_precomputed_table.pkl', 'rb') as file:
    results_each = pickle.load(file)

print(results_each['rand'][0])