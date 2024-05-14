from setuptools import setup, find_packages

setup(
    name="easyvizar-map",
    version="0.1",
    description="Manager for 3D mapping data",
    url="https://github.com/EasyVizAR/map/",

    project_urls = {
        "Homepage": "https://wings.cs.wisc.edu/easyvizar/",
        "Source": "https://github.com/EasyVizAR/map/",
    },

    packages=find_packages(),

    entry_points={
        "console_scripts": [
            "map = map.__main__:main"
        ]
    }
)
