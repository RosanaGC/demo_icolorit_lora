
## iColoriT LoRA (Demo)

Interfaz interactiva para colorizar imágenes en blanco y negro usando un modelo ViT con LoRA.

### Paso 1: pesos

- https://drive.google.com/file/d/1MXZFhYvQTRenU1xgjpYUIie4SjphY9Ky/view?usp=sharing
- Lora_r32_Dataset_4_epochs_1000_wES_lrf_1e-3.pth

### Paso 2: clonar repo

- git clone https://github.com/RosanaGC/demo_icolorit_lora.git
- cd demo_icolorit_lora


### Paso 3: entorno

#### Opción 1 - Virtual env

##### Crear entorno virtual
- python3 -m venv .venv
- source .venv/bin/activate

##### Instalar dependencias
- pip install -r requirements.txt

#### Opción 2 - Conda/Miniconda

#####  Instalar Miniconda
- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
- bash Miniconda3-latest-Linux-x86_64.sh
- cerrar y abrir terminal

##### Crear entorno
- conda env create -f environment.yml
- conda activate icolorit_demo

### Paso 4: ejecutar
- python3 icolorit_ui_v2.py --model_path <path/to/checkpoint.pth> --target_image <path/to/image>

### Quick Start (visual)
1. Abrí la app con un target (imagen a colorizar).
2. Hacé click en el Drawing Pad para agregar un hint.
3. También podés cargar hints desde `Load Hints` o con la tecla `H`.
4. Ajustá el color desde el gamut o la paleta.
5. Usá `+` / `-` para zoom en el Drawing Pad.
6. Guardá el resultado con `Save` o `Save As`.

Atajos útiles:
- `R`: reset
- `S`: save
- `Q`: save + quit
- `G`: toggle gray
- `L`: load nueva imagen
- `H`: load hints desde archivo

### Formatos soportados
- Target: `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`
- Reference: `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`

### Notas importantes
- `--model_path`: ruta al modelo.
- Si el modelo y el checkpoint no coinciden, la carga puede ser parcial (se usa `strict=False`).
- Formatos de hints soportados:
  - `.txt` / `.csv` / `.tsv`: una coordenada `x,y` o `x y` por línea, en espacio `224x224`.
  - `.json`: lista de pares `[x, y]`, lista en `points`, o el `hints.json` exportado por la app usando `hints[].pos_224`.

### Problemas comunes
- Si la app crashea al cargar la imagen, revisá que el archivo exista y sea legible por OpenCV.
- Si no ves cambios, probá con otra imagen de entrada y asegúrate de estar usando el `.pth` correcto.




#### Acknowledgments

Our GUI is an updated version of the [interactive-deep-colorization](https://github.com/junyanz/interactive-deep-colorization).
Thanks for sharing the codes!
