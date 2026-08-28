# iColoriT LoRA — Torres García

[Español](#español) | [English](#english)

## Español

Demo de una interfaz interactiva para agregar color a imágenes en **escala de
grises** mediante un modelo Vision Transformer (ViT) adaptado con LoRA.

El modelo fue ajustado con un conjunto de imágenes de obras de **Joaquín Torres
García**. El objetivo es asistir la exploración cromática de reproducciones en
escala de grises, tomando como referencia los colores y las relaciones visuales
aprendidas de ese conjunto. La aplicación genera una propuesta inicial y permite
guiar el resultado mediante puntos de color y una imagen de referencia.

> Esta herramienta produce una interpretación asistida por IA. El resultado no
> debe considerarse una reconstrucción histórica de los colores originales de
> una obra de Torres García.

## Instalación

### Paso 1: descargar los pesos

Descargar el archivo
`Lora_r32_Dataset_4_epochs_1000_wES_lrf_1e-3.pth` desde
[Google Drive](https://drive.google.com/file/d/1MXZFhYvQTRenU1xgjpYUIie4SjphY9Ky/view?usp=sharing).

### Paso 2: clonar el repositorio

```bash
git clone https://github.com/RosanaGC/demo_icolorit_lora.git
cd demo_icolorit_lora
```

### Paso 3: entorno

#### Opción 1 - Virtual env

Crear y activar el entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Opción 2 - Conda/Miniconda

Instalar Miniconda en Linux:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Cerrar y volver a abrir la terminal. Luego crear el entorno:

```bash
conda env create -f environment.yml
conda activate icolorit_demo
```

#### Instalar las dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: ejecutar

```bash
python3 icolorit_ui_v2.py
```

Al abrir la aplicación, cargar como *target* la reproducción en escala de
grises que se desea colorizar. De forma opcional, se puede cargar una imagen a
color como referencia para consultar su paleta y orientar los puntos de color.

### Formatos soportados

- Imagen objetivo (*target*): `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`
- Imagen de referencia: `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`

## Agradecimientos

Nuestra interfaz es una versión actualizada de
[interactive-deep-colorization](https://github.com/junyanz/interactive-deep-colorization).
¡Gracias por compartir el código!

---

## English

Demo of an interactive interface for adding color to **grayscale images** using
a Vision Transformer (ViT) model adapted with LoRA.

The model was fine-tuned on a dataset of images of works by **Joaquín Torres
García**. Its purpose is to support the chromatic exploration of grayscale
reproductions using colors and visual relationships learned from that dataset.
The application generates an initial proposal and lets the user guide the
result with color points and a reference image.

> This tool produces an AI-assisted interpretation. The result should not be
> considered a historical reconstruction of the original colors of a work by
> Torres García.

### Installation

#### Step 1: download the weights

Download
`Lora_r32_Dataset_4_epochs_1000_wES_lrf_1e-3.pth` from
[Google Drive](https://drive.google.com/file/d/1MXZFhYvQTRenU1xgjpYUIie4SjphY9Ky/view?usp=sharing).

#### Step 2: clone the repository

```bash
git clone https://github.com/RosanaGC/demo_icolorit_lora.git
cd demo_icolorit_lora
```

#### Step 3: set up the environment

##### Option 1 — Virtual environment

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

##### Option 2 — Conda/Miniconda

Install Miniconda on Linux:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Close and reopen the terminal, then create the environment:

```bash
conda env create -f environment.yml
conda activate icolorit_demo
```

##### Install the dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: run the application

```bash
python3 icolorit_ui_v2.py
```

After opening the application, load the grayscale reproduction to be colorized
as the *target* image. Optionally, load a color image as a reference to inspect
its palette and guide the color points.

#### Supported formats

- Target image: `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`
- Reference image: `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`

### Acknowledgments

Our GUI is an updated version of
[interactive-deep-colorization](https://github.com/junyanz/interactive-deep-colorization).
Thanks for sharing the code!
