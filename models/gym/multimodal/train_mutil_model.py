# %%
import wandb
wandb.login() 
# # 8599fbb702cb5767e13d2ac3b1cdcc1c9b65d451

# %%
from tem_dataloader import MultimodalDataset
environment_name = 'AmericanDinerExposure'

my_dataset = MultimodalDataset(f'/media/tyz/3B6FFE7354FF3296/11_777/tartanairv2filtered/{environment_name}/Data_easy')

# %%
from torch.utils.data import DataLoader, Dataset
batch_size = 32
train_dataloader = DataLoader(my_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

# %%
from custom_models.CustomViT import CustomViT
from custom_models.CustomViTMAE import CustomViTMAE
import torch
# call CustomViT
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEConfig
from PIL import Image

output_dir='/home/tyz/Desktop/11_777/camelmera/weights'

# Initialize a new CustomViTMAE model
model_name = "facebook/vit-mae-base"

vit_config = ViTMAEConfig.from_pretrained(model_name)
vit_config.output_hidden_states=True
vit_model = CustomViT.from_pretrained(model_name,config=vit_config)


model_name = "facebook/vit-mae-base"

config = ViTMAEConfig.from_pretrained(model_name)
config.output_hidden_states=True

# load from pretrained model and replace the original encoder with custom encoder
custom_model = CustomViTMAE.from_pretrained("facebook/vit-mae-base",config=config)
custom_model.vit = vit_model

# Load the state_dict from the saved model
state_dict = torch.load(f"{output_dir}/pytorch_model.bin")

# Apply the state_dict to the custom_model
custom_model.load_state_dict(state_dict)

# %%
import time
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in dataloader:
        # print(batch.keys())
        pixel_values = batch["pixel_values"].to(device)
        pixel_values1 = batch["pixel_values1"].to(device)
        pixel_values2 = batch["pixel_values2"].to(device)
        # batch = {k: v.to(device) for k, v in batch.items()}

        # bf_forward = time.time()

        optimizer.zero_grad()
        outputs = model(pixel_values,pixel_values1,pixel_values2,noise=None)

        # af_forward = time.time()
        # print("forward time: ", af_forward - bf_forward)

        loss = outputs.loss
        loss.backward()

        # af_backward = time.time()
        # print("backward time: ", af_backward - af_forward)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # af_step = time.time()
        # print("step time: ", af_step - af_backward)

        total_loss += loss.item() * pixel_values.size(0)
        total_samples += pixel_values.size(0)

        if total_samples % (batch_size * 1) == 0: # log every 1 batches
          wandb.log({'loss': loss.item()} )
          print(loss.item())
        
        # af_loss_cal = time.time()
        # print("loss cal time: ", af_loss_cal - af_step)

    return total_loss / total_samples


# %%
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

epochs = 1
learning_rate = 0.01
weight_decay = 0.01
warmup_steps = 0
total_steps = len(train_dataloader) * epochs
optimizer = AdamW(custom_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
custom_model.to(device)


# %%
wandb.init(project="11777",name=environment_name+time.strftime("%Y%m%d-%H%M%S"))
for epoch in range(epochs):
    train_loss = train(custom_model, train_dataloader, optimizer, scheduler, device)
    print(f"Epoch: {epoch + 1}, Loss: {train_loss:.4f}")
    output_dir='/home/tyz/Desktop/11_777/camelmera/weights'
    custom_model.save_pretrained(output_dir)
wandb.finish()

# %%
output_dir='/home/tyz/Desktop/11_777/camelmera/weights'
custom_model.save_pretrained(output_dir)

# %%



