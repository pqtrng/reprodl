FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel AS env

# Set the working directory
WORKDIR /reprodl

# Copy code and data
COPY requirements.txt .

# Install all libraries
RUN apt-get update && apt-get upgrade -y
RUN pip install -r requirements.txt
RUN apt-get install -y libsndfile1-dev vim

FROM env

# Copy code and data
COPY . ./

# Run the training loop
# CMD ["python", "train.py", "~trainer.gpus"]
CMD ["python", "train.py"]