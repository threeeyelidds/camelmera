# camelmera

## Encoder

### Prerequisites

Before running the training script for encoder, you must have the following:

- Python 3.x installed on your machine.
- tartanairv2filtered datasets downloaded on your machine's `/mnt/data/tartanairv2filtered` folder.

### Setup

1. Clone or download the repository.
2. Install the required Python packages using the following command, `requirements.txt` is under the `models/gym/multimodal` folder:

```
pip install -r models/gym/multimodal/requirements.txt
```

## Running the Trainning script

Using the following command to train multimodal encoder.

```
python models/gym/train_mutil_model_encoder.py
```

Using the following command to train unimodal encoder.

```
python models/gym/train_unimodel.py
```

If wandb is shown, use the token: `8599fbb702cb5767e13d2ac3b1cdcc1c9b65d451`.

The weights of the model would be saved at `/home/ubuntu/weights`
