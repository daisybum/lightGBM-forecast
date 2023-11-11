FROM arm64v8/debian:bullseye-slim
WORKDIR /app
RUN apt-get update
RUN apt-get install -y wget
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh

RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN ~/miniconda3/bin/conda init bash

# pip로 설치할 패키지들을 여기다 추가해주세요.
COPY ./requirements.txt ./requirements.txt

RUN /root/miniconda3/bin/pip install python-dotenv
RUN /root/miniconda3/bin/pip install -r requirements.txt
COPY . .

# python으로 구동할 코드를 연결해주세요.
ENTRYPOINT ["/root/miniconda3/bin/python", "/app/example.py"]