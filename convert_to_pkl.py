import json
import pickle
import time
import os

json_path = "model.json"
pkl_path = "model.pkl"

if not os.path.exists(json_path):
    print(f"Error: {json_path} not found.")
    exit(1)

print(f"Loading {json_path}... this takes 1-3 minutes.")
start = time.time()
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded JSON in {time.time() - start:.2f} seconds.")

print(f"Saving to {pkl_path}... this is fast.")
start = time.time()
with open(pkl_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved Pickle in {time.time() - start:.2f} seconds.")
print("Done! You can now use model.pkl.")
