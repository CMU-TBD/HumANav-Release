#!/bin/bash
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

#able to exit when failing to install a package
set -e

#checking that pip/conda are installed
which pip || (echo -e "${RED}pip not installed${NC}" && exit)
which conda || (echo -e "${RED}conda not installed${NC}" && exit)
# need to install tf 1.10.1 which is no longer maintained via this link:
#pip install https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.10.1.tar.gz
#install pip packages
PIP_PACKAGES="matplotlib dotmap pyassimp scikit-image scikit-fmm numpy==1.14.5"
echo -e "${CYAN}Using pip to install $PIP_PACKAGES ${NC}"
for i in $PIP_PACKAGES; do
    pip install $i || (echo -e "${RED}Failed to install $i${NC}" && exit 1)
    echo -e "${GREEN}Installed $i${NC}"
done

#install conda packages
CONDA_PACKAGES="opencv pyopengl pandas"
echo -e "${CYAN}Using conda to install $CONDA_PACKAGES ${NC}"
for i in $CONDA_PACKAGES; do
    conda install -y $i || (echo -e "${RED}Failed to install $i${NC}" && exit 1)
    echo -e "${GREEN}Installed $i${NC}"
done
echo -e "${GREEN}All packages installed! ${NC}"

