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

- Train model

```bash
python train.py
```

- Without GPU

```bash
python train.py ~trainer.gpus
```

- If you want to use Docker then:

- For training with GPU, change this line in Dockerfile:

```Docker
CMD ["python", "train.py"]
```

- Without GPU

```Docker
CMD ["python", "train.py", "~trainer.gpus"]
```

- Build

```bash
docker build . -t reprodl --rm
```

- Run

```bash
docker run --name "reprodl" reprodl
```
