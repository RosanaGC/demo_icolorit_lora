
### OPCION 1- Virtual env

# iColoriT Demo Software # 1. Clonar el repo
git clone https://github.com/RosanaGC/demo_icolorit_lora.git](https://github.com/RosanaGC/demo_icolorit_lora.git
cd demo_icolorit_lora

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Descargar checkpoints
bash scripts/download_checkpoints.sh

# 5. Ejecutar
python -m gui.gui_main --device auto


###  OPCION 2 CONDA/MINICONDA

# 1. Instalar Miniconda (una vez)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# cerrar y abrir terminal

# 2. Clonar el repo
git clone https://github.com/<user>/iColoriT_demo_2.git
cd iColoriT_demo_2

# 3. Crear entorno
conda env create -f environment.yml
conda activate icolorit

# 4. Descargar checkpoints
bash scripts/download_checkpoints.sh

# 5. Ejecutar
bash scripts/run_gui.sh --device auto

### OPCION 3 DOCKER

# 1. Clonar el repo
git clone https://github.com/<user>/iColoriT_demo_2.git
cd iColoriT_demo_2

# 2. Construir imagen (CPU)
docker build -t icolorit:cpu -f Dockerfile.cpu .

# 3. Correr contenedor (monta tu carpeta local)
docker run --rm -it -v $(pwd):/app icolorit:cpu



## Acknowledgments

Our GUI is an updated version of the [interactive-deep-colorization](https://github.com/junyanz/interactive-deep-colorization).
Thanks for sharing the codes!
