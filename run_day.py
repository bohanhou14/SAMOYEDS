import pickle

with open("profiles-agent_num=10-top_p=0.7-temp=2.0.pkl", "rb") as f:
    # a list of dictionaries
    profiles = pickle.load(f)

