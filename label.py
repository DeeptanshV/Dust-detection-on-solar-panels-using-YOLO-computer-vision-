import os

with open("labels.txt", "w") as f:
    for label, folder in enumerate(["Clean", "Dusty"]):
        for file in os.listdir(f"dataset/{folder}"):
            f.write(f"{folder}/{file},{label}\n")
