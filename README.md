# EasyVizAR Map Manager

# Installation

## Installation of the Snap Package

The snap package is automatically built and released to the snap store on each
commit to the main branch. It is currently built for amd64 and arm64 platforms.

Install the stable version:

    sudo snap install easyvizar-map

Install the latest build:

    sudo snap install --edge easyvizar-map

If this will be running on the same machine as the easyvizar-edge snap
(recommended), you should connect the easyvizar-edge:data interface to allow
easyvizar-detect to read image files stored by easyvizar-edge.

    sudo snap connect easyvizar-map:data easyvizar-edge:data
