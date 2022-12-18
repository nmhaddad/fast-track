FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt update

RUN apt install -y zip htop screen libgl1-mesa-glx

RUN pip install seaborn thop

RUN git clone https://github.com/WongKinYiu/yolov7.git

RUN cd yolov7

RUN pip install -r requirements.txt