import os

folders = {
    "Train X-Ray": "data/dataset/train/x-ray",
    "Val X-Ray": "data/dataset/val/x-ray",
    "Train Non-X-Ray": "data/dataset/train/non_x-ray",
    "Val Non-X-Ray": "data/dataset/val/non_x-ray",
}

print("--- File Counts ---")
total = 0
for label, path in folders.items():
    if os.path.exists(path):
        # Count non-hidden files
        count = len([f for f in os.listdir(path) if not f.startswith('.')])
        print(f"{label:<16}: {count} files")
        total += count
    else:
        print(f"{label:<16}: Folder does not exist!")

print("-" * 25)
print(f"Total Dataset Files: {total}")