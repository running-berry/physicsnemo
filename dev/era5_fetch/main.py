import yaml
from era5_download import era5_download

def main():
    with open("era5.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    era5_download(cfg)

if __name__ == "__main__":
    main()
