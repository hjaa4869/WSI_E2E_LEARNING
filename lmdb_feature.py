import lmdb
import torch
import numpy as np
import pickle
from PIL import Image
import io
import torchvision.transforms as transforms
from models.resnet_custom import resnet50_baseline

# ===== model =====
model = resnet50_baseline(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ===== LMDB =====
input_path = "lmdb"
output_path = "lmdb_res50"

env_in = lmdb.open(input_path, readonly=True, lock=False)
env_out = lmdb.open(output_path, map_size=1e12)

# ===== transform =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

batch_size = 32
batch_imgs = []
batch_keys = []

with env_in.begin() as txn_in:
    cursor = txn_in.cursor()
    txn_out = env_out.begin(write=True)

    for idx, (key, value) in enumerate(cursor):

        try:
            # ===== 1. decode LMDB value =====
            img_bytes = pickle.loads(value)

            # ===== 2. bytes → PIL =====
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # ===== 3. transform =====
            img_tensor = transform(img)

            batch_imgs.append(img_tensor)
            batch_keys.append(key)

        except Exception as e:
            print(f"skip {key}: {e}")
            continue

        # ===== 4. batch inference =====
        if len(batch_imgs) == batch_size:
            batch_tensor = torch.stack(batch_imgs).to(device)

            with torch.no_grad():
                feats = model(batch_tensor)

            feats = feats.squeeze(-1).squeeze(-1).cpu().numpy()

            # ===== 5. write LMDB =====
            for k, f in zip(batch_keys, feats):
                txn_out.put(k, pickle.dumps(f.astype(np.float32)))

            txn_out.commit()
            txn_out = env_out.begin(write=True)

            batch_imgs, batch_keys = [], []

    # ===== last batch =====
    if batch_imgs:
        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            feats = model(batch_tensor)

        feats = feats.squeeze(-1).squeeze(-1).cpu().numpy()

        for k, f in zip(batch_keys, feats):
            txn_out.put(k, pickle.dumps(f.astype(np.float32)))

        txn_out.commit()

env_in.close()
env_out.close()