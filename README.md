
## paso 1
### pesos

https://drive.google.com/file/d/1MXZFhYvQTRenU1xgjpYUIie4SjphY9Ky/view?usp=sharing
gdown "https://drive.google.com/uc?id=1MXZFhYvQTRenU1xgjpYUIie4SjphY9Ky" -O Lora_r32_Dataset_4_epochs_1000_wES_lrf_1e-3.pth

## paso 2 iColoriT Demo Software 

git clone https://github.com/RosanaGC/demo_icolorit_lora.git
cd demo_icolorit_lora


## OPCION 1- Virtual env

Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

### Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

### Ejecutar



##  OPCION 2 CONDA/MINICONDA

### 1. Instalar Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
### 2 cerrar y abrir terminal

### 3. Crear entorno
conda env create -f environment.yml
conda activate icolorit

### 4. Ejecutar





## Acknowledgments

Our GUI is an updated version of the [interactive-deep-colorization](https://github.com/junyanz/interactive-deep-colorization).
Thanks for sharing the codes!
