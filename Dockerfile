FROM pytorch/pytorch:latest

RUN pip install --upgrade pip

RUN pip install numpy pandas opencv-python matplotlib
RUN pip install ultralytics
#dependencies for opencv and yolov8
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#sklearn
RUN pip install scikit-learn
# Set the default working directory
WORKDIR /app

# # Copy your course files to the working directory
# COPY . /app

# Run a command in the background that will keep the container running
CMD while true; do echo "Hello, world!"; sleep 1; done