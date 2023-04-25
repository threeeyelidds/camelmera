# define CustomViTMAE, add two decoders, use combined loss
# 3 masks & 3 id_stores for 3 modalities
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder,ViTMAEForPreTrainingOutput
from transformers import ViTMAEForPreTraining
import torch.utils.data
import torch

class CustomViTMAE(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # replace the original vit
        # self.vit = CustomViT(config)

        # define the decoder layers similar to ViTMAEForPreTraining
        self.decoder1 = ViTMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)
        self.decoder2 = ViTMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)

        # load the pretrained weights of the decoder layers from ViTMAEForPreTraining
        self.decoder1.load_state_dict(self.decoder.state_dict())
        self.decoder2.load_state_dict(self.decoder.state_dict())

        self.post_init()

    def forward(self, pixel_values, pixel_values1, pixel_values2, noise=None, head_mask=None, output_attentions=None, output_hidden_states=None,return_dict=None, **kwargs):
        # need to compute pixel_values and mask for each modality
        # original code for ViTMAEForPreTraining.forward
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            pixel_values1,
            pixel_values2,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        latent = outputs.last_hidden_state
        ids_restore,ids_restore1,ids_restore2 = outputs.ids_restore
        mask,mask1,mask2 = outputs.mask # mask comes from ViTMAEEmbeddings

        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        combined_loss = self.forward_loss(pixel_values, logits, mask)

        decoder_outputs1 = self.decoder1(latent, ids_restore)
        logits1 = decoder_outputs1.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        combined_loss += self.forward_loss(pixel_values1, logits1, mask1)

        decoder_outputs2 = self.decoder2(latent, ids_restore)
        logits2 = decoder_outputs2.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        combined_loss += self.forward_loss(pixel_values2, logits2, mask2)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((combined_loss,) + output) if combined_loss is not None else output

        return ViTMAEForPreTrainingOutput(
            loss=combined_loss,
            logits=logits,
            mask=(mask,mask1,mask2),
            ids_restore=(ids_restore,ids_restore1,ids_restore2),
            hidden_states=latent,
            attentions=outputs.attentions,
        )


        # # decode images with the three decoders
        # decoded_pixels0 = self.decoder(last_hidden_state)
        # decoded_pixels1 = self.decoder1(last_hidden_state)
        # decoded_pixels2 = self.decoder2(last_hidden_state)
        

        # # compute the reconstruction losses for each decoder
        # reconstruction_loss0 = compute_custom_reconstruction_loss(decoded_pixels0, pixel_values)
        # reconstruction_loss1 = compute_custom_reconstruction_loss(decoded_pixels1, pixel_values)
        # reconstruction_loss2 = compute_custom_reconstruction_loss(decoded_pixels2, pixel_values)

        # # compute the combined reconstruction loss
        # combined_loss = reconstruction_loss1 + reconstruction_loss2 + reconstruction_loss0

        # # add the combined loss to the original loss
        # outputs.loss += combined_loss

        # return outputs
