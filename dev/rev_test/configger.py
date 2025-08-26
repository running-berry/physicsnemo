from ruamel.yaml import YAML
from pathlib import Path

class ConfigProcesser(YAML):
    def __init__(self):
        super().__init__(typ='rt')
        self.preserve_quotes = True
        self.indent(mapping=2, sequence=4, offset=2)
        self.default_flow_style = False

    def load(self, path):
        path = Path(path)
        with path.open() as f:
            return super().load(f)

    def dump(self, data, path):
        path = Path(path)
        with path.open("w") as f:
            super().dump(data, f)  

