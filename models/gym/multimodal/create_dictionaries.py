
from tem_dataloader import MultimodalDataset, MultimodalDatasetPerTrajectory
import pickle

from transformers import ViTMAEForPreTraining, ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel
import torch
# environment_name = 'AmericanDinerExposure'
environment_name = 'AbandonedCableExposure'
# my_dataset = MultimodalDataset(f'/media/tyz/3B6FFE7354FF3296/11_777/tartanairv2filtered/{environment_name}/Data_easy')
import os
import sys
from pympler import asizeof
# model_bin_dir='/Users/henryxu_1/Desktop/Carnegie\ Mellon\ University/Spring\ 2023/11-777/Group\ Project/weights'
# model_bin_dir = os.path.join('/Users/henryxu_1/Desktop', 'Carnegie Mellon University', 'Spring 2023', '11-777', 'Group Project', 'weights')
model_bin_dir = '/home/tyz/Desktop/11_777/camelmera/weights'
# environemnt_directory = os.path.join('/Users/henryxu_1/Desktop', 'Carnegie Mellon University', 'Spring 2023', '11-777', 'Group Project', environment_name, 'Data_easy')
environemnt_directory = f'/media/tyz/3B6FFE7354FF3296/11_777/tartanairv2filtered/{environment_name}/Data_easy'
# output_dir = f'/Users/henryxu_1/Desktop/{environment_name}'
output_dir = f'/home/tyz/Desktop/11_777/{environment_name}' # make sure to change this for RL training
os.makedirs(output_dir, exist_ok=True)
for folder in os.listdir(environemnt_directory):
    trajectory_folder_path = os.path.join(environemnt_directory, folder)
    if not os.path.isdir(trajectory_folder_path):
        continue

    my_dataset = MultimodalDatasetPerTrajectory(trajectory_folder_path)

    from torch.utils.data import DataLoader, Dataset
    batch_size = 32
    train_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False)

    from custom_models.CustomViT import CustomViT
    from custom_models.CustomViTMAE import CustomViTMAE
    import torch
    # call CustomViT
    from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEConfig
    from PIL import Image

    # output_dictionary = {'embedding': [], 'goals': []}


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
    custom_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    custom_model.to(device)

    # create Unimodel ViT
    unimodal_model_name = "facebook/vit-mae-base"
    unimodal_vit_config = ViTMAEConfig.from_pretrained(unimodal_model_name)
    unimodal_vit_config.output_hidden_states=True
    unimodal_vit_model = ViTMAEModel.from_pretrained(unimodal_model_name, config=unimodal_vit_config)
    unimodal_vit_model.eval()

    # Load the state_dict from the saved model
    state_dict = torch.load(os.path.join(model_bin_dir, "pytorch_model.bin"), map_location=torch.device('cpu'))

    # Apply the state_dict to the custom_model
    custom_model.load_state_dict(state_dict)

    for index, batch in enumerate(train_dataloader):
        output_dictionary = {'embedding': [], 'goals': []}
        # output_list = [[],[]]
        # print(batch.keys())
        pixel_values = batch["pixel_values"].to(device)
        pixel_values1 = batch["pixel_values1"].to(device)
        pixel_values2 = batch["pixel_values2"].to(device)
        outputs = custom_model(pixel_values,pixel_values1,pixel_values2,noise=None)
    # hidden_states[:,0,:] (1,1,768)
        # print(outputs.hidden_states.shape)
        # output_dictionary["embedding"].append(outputs.hidden_states[:,0,:])

        # goal = batch[-1]
        for element_index in range(pixel_values.shape[0]):
            # print(f"this is the size of the embeddings: {outputs.hidden_states[element_index,0,:].shape}")
            output_dictionary["embedding"].append(outputs.hidden_states[element_index,0,:])
            # output_list[0].append(outputs.hidden_states[element_index,0,:])

        # print(f"This is pixel value shape: {pixel_values.shape}")
        # unimodal output to save as goal later
        unimodal_outputs = unimodal_vit_model(pixel_values[-1,:,:,:].unsqueeze(0))
        # print(f"unimodal output shape: {unimodal_outputs.hidden_states}")
        # print(f"unimodal output shape: {unimodal_outputs.last_hidden_state.shape}")
        output_dictionary["goals"].append(unimodal_outputs.last_hidden_state[0,0,:])
        # output_list[1].append(unimodal_outputs.last_hidden_state[0,0,:])
        trajectory_folder_dir = f"{output_dir}/{folder}"
        os.makedirs(trajectory_folder_dir, exist_ok=True)

        # print(f"sys get size of embeddings: {sys.getsizeof(output_list[0])}")
        # print(f"sys get size of goals: {sys.getsizeof(output_list[1])}")
        # print(f"asizeof get size of embeddings: {asizeof.asizeof(output_list[0])}")
        # print(f"asizeof get size of embeddings: {asizeof.asizeof(output_list[0])}")

        with open(f'{trajectory_folder_dir}/{index}.pkl', 'wb') as file:
            pickle.dump(output_dictionary, file)
            # pickle.dump(output_list, file)






# # %%
# import time
# def train(model, dataloader, optimizer, scheduler, device):
#     model.train()
#     total_loss = 0
#     total_samples = 0
    
#     for batch in dataloader:
#         # print(batch.keys())
#         pixel_values = batch["pixel_values"].to(device)
#         pixel_values1 = batch["pixel_values1"].to(device)
#         pixel_values2 = batch["pixel_values2"].to(device)
#         # batch = {k: v.to(device) for k, v in batch.items()}

#         # bf_forward = time.time()

#         optimizer.zero_grad()
        # outputs = model(pixel_values,pixel_values1,pixel_values2,noise=None)

#         # af_forward = time.time()
#         # print("forward time: ", af_forward - bf_forward)

#         loss = outputs.loss
#         loss.backward()

#         # af_backward = time.time()
#         # print("backward time: ", af_backward - af_forward)

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         scheduler.step()

#         # af_step = time.time()
#         # print("step time: ", af_step - af_backward)

#         total_loss += loss.item() * pixel_values.size(0)
#         total_samples += pixel_values.size(0)

#         if total_samples % (batch_size * 1) == 0: # log every 1 batches
#           wandb.log({'loss': loss.item()} )
#           print(loss.item())
        
#         # af_loss_cal = time.time()
#         # print("loss cal time: ", af_loss_cal - af_step)

#     return total_loss / total_samples


# # %%
# from transformers.optimization import AdamW
# from transformers import get_linear_schedule_with_warmup

# epochs = 1
# learning_rate = 0.01
# weight_decay = 0.01
# warmup_steps = 0
# total_steps = len(train_dataloader) * epochs
# optimizer = AdamW(custom_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# custom_model.to(device)


# # %%
# wandb.init(project="11777",name=environment_name+time.strftime("%Y%m%d-%H%M%S"))
# for epoch in range(epochs):
#     train_loss = train(custom_model, train_dataloader, optimizer, scheduler, device)
#     print(f"Epoch: {epoch + 1}, Loss: {train_loss:.4f}")
#     output_dir='/home/tyz/Desktop/11_777/camelmera/weights'
#     custom_model.save_pretrained(output_dir)
# wandb.finish()

# # %%
# output_dir='/home/tyz/Desktop/11_777/camelmera/weights'
# custom_model.save_pretrained(output_dir)

# # %%



