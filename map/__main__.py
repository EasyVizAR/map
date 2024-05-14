import os

from .mapperclient import MapperClient


EASYVIZAR_SERVER = os.environ.get("EASYVIZAR_SERVER", "http://localhost:5000")

SEED_LOCATION_ID = "8a58613d-f207-44dd-8f61-effaea9abde6"


def main():
    client = MapperClient(EASYVIZAR_SERVER, SEED_LOCATION_ID)
    client.run()


if __name__ == "__main__":
    main()
