import pickle

    # Replace 'your_file.pkl' with the actual path to your PKL file
with open('/Users/ethan/Documents/BURise/modeling_project/bdnf_network_analysis/bdnf_analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)