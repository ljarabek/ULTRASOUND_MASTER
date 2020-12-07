import json
import pickle

with open("./files/train/-2419462809323158939", "rb") as file:
    arr = pickle.load(file)

print(arr)