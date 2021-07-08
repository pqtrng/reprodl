# Reproducible Deep Learning

A simple deep learning model implemented in a notebook, and port it to a ‘reproducible’ world by including code versioning (**Git**), data versioning (**DVC**), experiment logging (**Weight & Biases**), hyper-parameter tuning, configuration (**Hydra**), and **‘Dockerization’**.
This is from the course of Prof. [**Simone Scardapane**](https://www.sscardapane.it/)

- Setup your machine

```bash
conda create -n reprodl
conda activate reprodl
```

- Install generic prerequisites

```bash
pip install -r requirements.txt
```

- Download dataset [ESC-50](https://github.com/karolpiczak/ESC-50)

- Extract zip file to **data** folder
- Run wandb

```bash
docker pull wandb/local
wandb local
```

- Train model

```bash
python train.py
```

- Without GPU

```bash
python train.py ~trainer.gpus
```

- If you want to use Docker then:

1. Build

```bash
docker build . -t reprodl --rm
```

2. Run

```bash
docker run --name "reprodl" -it reprodl
```

3. Train:

```bash
python train.py 
```

- Without GPU

```bash
python train.py ~trainer.gpus
```
