from lightning.pytorch.cli import LightningCLI

from src.model import RWKV
from src.data import RWKVDataModule

def cli_main():
    LightningCLI(RWKV, RWKVDataModule, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    cli_main()
