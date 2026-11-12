# EuroSAT Land-Cover Classifier

An original, self-contained computer-vision project that classifies 64×64 Sentinel-2 RGB satellite patches into 10 land-cover categories. It includes a reproducible stratified split, an in-house compact CNN, training/evaluation, per-class metrics, confusion matrix, Grad-CAM visualization, and single-image prediction.

## Setup

Python 3.10 or newer is recommended. Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download and verify the official RGB archive (about 95 MB), then extract it under `data/`:

```bash
python download_data.py
```

The expected folder is `data/EuroSAT_RGB/` with one subfolder for each class. The dataset authors identify the dataset as MIT-licensed; retain its attribution and follow the [Copernicus Sentinel Data Terms](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice). See the [Zenodo record](https://zenodo.org/records/7711810) and [official dataset repository](https://github.com/phelber/EuroSAT).

## Train and evaluate

```bash
python train.py --data data/EuroSAT_RGB --epochs 8 --batch-size 64
```

Use `--epochs` to change training duration and `--seed` to reproduce the deterministic split. The first run creates `outputs/` with the checkpoint, split manifest, metrics JSON, confusion matrix, and training curve. CPU is supported; training takes longer than on a CUDA GPU. The prediction command below creates a Grad-CAM overlay.

For a shorter balanced local run, cap the training pool at 500 examples per class: `python train.py --data data/EuroSAT_RGB --epochs 5 --batch-size 64 --train-per-class 500 --threads 4 --cpu`. The test and validation splits still use all their images. Omit the cap for full training.

## Predict and explain

```bash
python predict.py path/to/satellite_patch.jpg --checkpoint outputs/best_model.pt
```

To open the notebook, install Jupyter and run `jupyter lab EuroSAT_Project.ipynb`. In Colab, upload this folder or open the notebook from Drive; set the runtime to GPU for faster training. The notebook downloads data on first run and calls the same scripts as the command line.

## Project boundaries

The model is trained from scratch and uses RGB bands only. EuroSAT also has a 13-band multispectral archive, which this implementation does not use. Results in a run belong to that run only; they are not copied from published benchmarks. The random image-level split can place visually related patches in different splits, so accuracy should be interpreted as a benchmark result rather than a guarantee of geographic generalization.

## Attribution

Dataset and base paper: Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). “EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.” *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7), 2217–2226. [Paper](https://arxiv.org/abs/1709.00029) · [Dataset DOI](https://doi.org/10.5281/zenodo.7711810).

The training pipeline and model code in this folder are independently written for this project. A public EuroSAT implementation was consulted for workflow ideas and is cited as a reference in `SOURCES.md`; no files, trained weights, or reported metrics from it are included here.

## Classes

AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.
