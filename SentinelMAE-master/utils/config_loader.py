import yaml

def load_config(path="training/config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config
