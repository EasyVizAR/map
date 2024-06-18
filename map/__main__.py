import os
import pprint

from .mapperclient import MapperClient


EASYVIZAR_SERVER = os.environ.get("EASYVIZAR_SERVER", "http://localhost:5000")

SNAP_DATA = os.environ.get("SNAP_DATA")


def load_configuration():
    search_dir = os.environ.get("SNAP_DATA", "./")
    config_path = os.path.join(search_dir, "config.json")
    try:
        with open(config_path, "r") as source:
            config = json.load(source)
            return config
    except:
        pass

    return {}


def main():
    config = load_configuration()
    print("Loaded configuration:")
    pprint.pprint(config)

    client = MapperClient(EASYVIZAR_SERVER, config)
    client.run()


if __name__ == "__main__":
    main()
