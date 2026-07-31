import os
import urllib.request

train_dir = "data/dataset/train/non_x-ray"
val_dir = "data/dataset/val/non_x-ray"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print("Downloading 1,000 random non-X-ray images...")


opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')]
urllib.request.install_opener(opener)

for i in range(1, 1001):

    url = f"https://picsum.photos/seed/{i}/224/224"
    
    if i <= 800:
        save_path = os.path.join(train_dir, f"non_xray_{i}.jpg")
    else:
        save_path = os.path.join(val_dir, f"non_xray_{i}.jpg")
    
    urllib.request.urlretrieve(url, save_path)
    

    print(f"Downloaded {i}/1000 images...")

print("\nDone! All images saved to 'data/dataset/'")