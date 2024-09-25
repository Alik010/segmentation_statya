FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 gcc -y

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CMD [ "python", "trainmodel.py" ]
CMD ["bash"]
