import yaml

class Config:
    def __init__(self, config_dict):
        self.dataset_paths = config_dict['dataset_paths']
        self.rwrf = self.dataset_paths['rwrf']
        self.era5 = self.dataset_paths['era5']
        self.pptn = self.dataset_paths['pptn']

# Load once at startup
def load_config(config_path="./config.yaml"):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)

CONFIG = load_config()
