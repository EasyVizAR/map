import os

from .mapperclient import MapperClient


EASYVIZAR_SERVER = os.environ.get("EASYVIZAR_SERVER", "http://localhost:5000")


def main():
    client = MapperClient(EASYVIZAR_SERVER)
    client.run()


if __name__ == "__main__":
    main()
