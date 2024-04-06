import os
from collections import defaultdict

folder_path = './datasets/game/labels'

class_stats = defaultdict(int)

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        full_path = os.path.join(folder_path, filename)
        with open(full_path, 'r') as file:
            for line in file:
                class_id = line.split()[0]
                class_stats[class_id] += 1

for class_id, count in class_stats.items():
    print(f"Class: {class_id}: {count}")