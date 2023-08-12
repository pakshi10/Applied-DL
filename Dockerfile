FROM pytorch/pytorch:latest

RUN pip install --upgrade pip

RUN pip install numpy pandas opencv-python matplotlib

# Set the default working directory
WORKDIR /app

# # Copy your course files to the working directory
# COPY . /app

# Run a command in the background that will keep the container running
CMD while true; do echo "Hello, world!"; sleep 1; done