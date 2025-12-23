
### paso 1: pesos

- https://drive.google.com/file/d/1MXZFhYvQTRenU1xgjpYUIie4SjphY9Ky/view?usp=sharing
- Lora_r32_Dataset_4_epochs_1000_wES_lrf_1e-3.pth

### paso 2: clonar repo

- git clone https://github.com/RosanaGC/demo_icolorit_lora.git
- cd demo_icolorit_lora


### Paso 3: entorno
#### OPCION 1- Virtual env

##### Crear entorno virtual
- python3 -m venv .venv
- source .venv/bin/activate

##### Instalar dependencias
- pip install -r requirements.txt

####  OPCION 2 CONDA/MINICONDA

#####  Instalar Miniconda
- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
- bash Miniconda3-latest-Linux-x86_64.sh
- cerrar y abrir terminal

#####  crear entorno
- conda env create -f environment.yml
- conda activate icolorit_demo

### paso 3: ejecutar





#### Acknowledgments

Our GUI is an updated version of the [interactive-deep-colorization](https://github.com/junyanz/interactive-deep-colorization).
Thanks for sharing the codes!
