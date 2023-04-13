from transformers import ViTConfig
from models import MyModel

# define the model
config = ViTConfig.from_pretrained('facebook/vit-mae-base')
config.image_size = 224
model = MyModel(config)

# data loader
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch import Dataset

transform = Compose([
    Resize((config.image_size, config.image_size)),
    CenterCrop((config.image_size, config.image_size)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = [...]  # list of paths to your images
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_dir, self.image_paths[idx]))
        image = transform(image)
        return image

# define the optimizer and the learning rate scheduler
from transformers import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# train the model
from torch.utils.data import DataLoader

train_dataset = CustomDataset('path/to/train/images')
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

for epoch in range(10):
    for images in train_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    print(f'Epoch {epoch + 1}, loss: {loss.item():.5f}')

