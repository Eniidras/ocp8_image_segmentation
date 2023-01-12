# Base image
FROM python:3.10

# Set the working directory
WORKDIR /web-app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

RUN pip install -r requirements.txt

# These commands install the cv2 dependencies that are normally present on the local machine, but might be missing in your Docker container causing the issue.
RUN apt update

RUN apt install ffmpeg libsm6 libxext6  -y

COPY . .

CMD ["python", "main.py"]

EXPOSE 5000