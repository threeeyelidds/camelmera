
import wandb
wandb.login() 
# 8599fbb702cb5767e13d2ac3b1cdcc1c9b65d451

from multimodal.tem_dataloader import MultimodalDataset
trained_model_name = "unimodel_1"
environment_name = 'AbandonedCableExposure'

my_dataset = MultimodalDataset(f'/mnt/data/tartanairv2filtered/{environment_name}/Data_easy')

from torch.utils.data import DataLoader
batch_size = 32
train_dataloader = DataLoader(my_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

from transformers import ViTMAEForPreTraining, ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel
import torch

# create Unimodel ViT
model_name = "facebook/vit-mae-base"
vit_config = ViTMAEConfig.from_pretrained(model_name)
vit_config.output_hidden_states=True
vit_model = ViTMAEModel.from_pretrained(model_name,config=vit_config)

config = ViTMAEConfig.from_pretrained(model_name)
config.output_hidden_states=True

# load from pretrained model and replace the original encoder with custom encoder
unimodel = ViTMAEForPreTraining.from_pretrained(model_name,config=config)
unimodel.vit = vit_model

output_dir='/home/ubuntu/weights/' + trained_model_name

import os
if os.path.exists(output_dir):
    # Load the state_dict from the saved model
    state_dict = torch.load(f"{output_dir}/pytorch_model.bin")

    # Apply the state_dict to the custom_model
    unimodel.load_state_dict(state_dict)

from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

epochs = 1
learning_rate = 0.01
weight_decay = 0.01
warmup_steps = 0
total_steps = len(train_dataloader) * epochs
optimizer = AdamW(unimodel.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
unimodel.to(device)

import time
import tqdm
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm.tqdm(dataloader):
        if trained_model_name == "unimodel_0":
            pixel_values = batch["pixel_values"].to(device)
        elif trained_model_name == "unimodel_1":
            pixel_values = batch["pixel_values1"].to(device)
        elif trained_model_name == "unimodel_2":
            pixel_values = batch["pixel_values2"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values, noise=None)

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * pixel_values.size(0)
        total_samples += pixel_values.size(0)

        if total_samples % 32 == 0:
            wandb.log({'loss': loss.item()} )
            # print(loss.item())

    return total_loss / total_samples

wandb.init(project="11777",name=trained_model_name+"_"+environment_name+time.strftime("%Y%m%d-%H%M%S"))
for epoch in range(epochs):
    train_loss = train(unimodel, train_dataloader, optimizer, scheduler, device)
    print(f"Epoch: {epoch + 1}, Loss: {train_loss:.4f}")
    unimodel.save_pretrained(output_dir)
wandb.finish()

unimodel.save_pretrained(output_dir)