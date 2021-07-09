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

- Pull **Weights and Biases** image

```Docker
docker pull wandb/local
```

- Start **Weights and Biases** server

```bash
wandb local
```

- Finish validation step, and open [http://localhost:8080](http://localhost:8080) in your browser.

- Train model

```bash
python train.py
```

- Without GPU

```bash
python train.py ~trainer.gpus
```

- Tune model

```bash
python tune.py
```

- Run jobs

```bash
python scheduler.py
```

- If you want to use **Docker** then:

- Build

```bash
docker build . -t reprodl --rm
```

- Run

```bash
docker run --name "reprodl" reprodl -it
```
