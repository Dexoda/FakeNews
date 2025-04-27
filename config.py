import yaml

class Config:
    def __init__(self, path='config.yml'):
        with open(path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)
    
    def __getattr__(self, name):
        return self._data.get(name)

config = Config()
