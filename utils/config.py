import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    @property
    def project(self):
        return self.config['project']
    
    @property
    def data(self):
        return self.config['data']
    
    @property
    def model(self):
        return self.config['model']
    
    @property
    def training(self):
        return self.config['training']
    
    @property
    def api(self):
        return self.config['api']

# Global config instance
config = Config()
