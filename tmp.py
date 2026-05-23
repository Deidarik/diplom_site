import pickle

with open('web_model_data_v2.pkl', 'rb') as f:
    data = pickle.load(f)

print("Реальные ключи в файле v2:", list(data.keys()))