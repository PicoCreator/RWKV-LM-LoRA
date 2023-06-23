from lightning.pytorch.cli import LightningCLI

from src.model import RWKV
from src.data import get_data_module

# Locking the seed, for better replication of train runs
from lightning.fabric import seed_everything
seed_everything(3941088705)

def cli_main():
    LightningCLI(RWKV, get_data_module, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    cli_main()
