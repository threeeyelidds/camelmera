from tem_dataloader import MultimodalDataset
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MultimodalDataset('/media/tyz/3B6FFE7354FF3296/11_777/tartanairv2filtered/AbandonedCableExposure/Data_easy')

from torch.utils.data import DataLoader
batch_size = 32
train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

t_time = time.time()
for batch in train_dataloader:
    print(f"one batch time: {time.time() - t_time}")
    t_time = time.time()
    pixel_values = batch["pixel_values"].to(device)
    pixel_values1 = batch["pixel_values1"].to(device)
    pixel_values2 = batch["pixel_values2"].to(device)
    af_todevice = time.time()
    print("todevice time: ", af_todevice - t_time)
    pass