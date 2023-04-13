# define CustomViT, add two embedding layers for lidar and depth, add extra MLP before encoder
# note that the input should be three tensors, the output contains 3 masks & 3 id_stores for 3 modalities
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel, ViTMAEEncoder, ViTMAEEmbeddings, ViTMAEModelOutput
import torch.nn as nn

class CustomViT(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        hidden_size = 768
        
        # Three separate embeddings for the modalities
        self.embeddings1 = ViTMAEEmbeddings(config)
        self.embeddings2 = ViTMAEEmbeddings(config)

        # load the pretrained weights of the embeddings layers from ViTMAEModel
        self.embeddings1.load_state_dict(self.embeddings.state_dict())
        self.embeddings2.load_state_dict(self.embeddings.state_dict())
        
        # MLP to combine the modalities' embeddings
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.post_init()
    
    def forward(self, pixel_values, pixel_values1, pixel_values2, noise=None, head_mask=None, output_attentions=None,output_hidden_states=None, return_dict=None):
        '''
        input: a dict of inputs: {pixel_values, pixel_values1, pixel_values2}, each has torch.Size([1, 3, 224, 224])
        call it as model(**inputs)
        ''' 
        # original code
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Embeddings for three modalities
        embeddings, mask, ids_restore = self.embeddings(pixel_values,noise=noise)
        embeddings1, mask1, ids_restore1 = self.embeddings1(pixel_values1,noise=noise)
        embeddings2, mask2, ids_restore2 = self.embeddings2(pixel_values2,noise=noise)
        print(embeddings.shape)
        # Concatenate the embeddings
        concat_embeddings = torch.cat([embeddings, embeddings1, embeddings2], dim=2)
        print(concat_embeddings.shape)
        # Pass through the MLP
        conbined_embeddings = self.mlp(concat_embeddings)

        encoder_outputs = self.encoder(
            conbined_embeddings,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=(mask,mask1,mask2),
            ids_restore=(ids_restore,ids_restore1,ids_restore2),
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        # # Pass through the ViT Encoder
        # encoder_output = self.encoder(conbined_embeddings,head_mask=None,output_attentions=False,output_hidden_states=False,return_dict=True)
        
        # # Classification head
        # logits = self.classifier(encoder_output.last_hidden_state[:, 0])
        
        # if not return_dict:
        #     return logits, encoder_output.hidden_states, encoder_output.attentions
        # else:
        #     return {"logits": logits, "hidden_states": encoder_output.hidden_states, "attentions": encoder_output.attentions}
