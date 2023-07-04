from lightning.pytorch import Trainer

#
# We extend the 
class RWKVLightningTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)