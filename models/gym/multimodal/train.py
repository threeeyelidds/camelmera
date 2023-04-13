from dataloader import Dataloader
from custom_models.CustomViT import CustomViT
from custom_models.CustomViTMAE import CustomViTMAE
import torch
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEConfig

# load the dataset
dataset = Dataloader.load_dataset('dataset_1.pkl')

# call CustomViT
model_name = "facebook/vit-mae-base"
vit_config = ViTMAEConfig.from_pretrained(model_name)
vit_config.output_hidden_states=True
vit_model = CustomViT.from_pretrained(model_name,config=vit_config)

config = ViTMAEConfig.from_pretrained(model_name)
config.output_hidden_states=True

# load from pretrained model and replace the original encoder with custom encoder
custom_model = CustomViTMAE.from_pretrained("facebook/vit-mae-base",config=config)
custom_model.vit = vit_model


inputs = image_processor(images=image, return_tensors="pt")
inputs['pixel_values1']=inputs.pixel_values
inputs['pixel_values2']=inputs.pixel_values



# check forward is fine
outputs = vit_model(**inputs)
mask,mask1,mask2 = outputs.mask
ids_restore,ids_restore1,ids_restore2 = outputs.ids_restore
hidden_states=outputs.hidden_states