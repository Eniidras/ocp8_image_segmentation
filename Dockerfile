# Base image
FROM python:3.6

# Set the working directory
WORKDIR /web-app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

RUN pip install -r requirements.txt

# These commands install the cv2 dependencies that are normally present on the local machine, but might be missing in your Docker container causing the issue.
RUN apt update

RUN apt install ffmpeg libsm6 libxext6  -y

COPY . .

CMD ["python3", "main.py", "host", "0.0.0.0", "--port", "5000"]

EXPOSE 5000