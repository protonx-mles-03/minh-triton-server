# Base image and infos
FROM nvcr.io/nvidia/tritonserver:23.05-py3
LABEL maintainer="nhatminh.cdt@gmail.com" \
      description="Dockerfile containing all the requirements for text detection model" \
      version="1.0"

# Prepare environment
WORKDIR /srv
ADD ./requirements.txt ./requirements.txt
ADD model_repository ./model_repository

# Install requirements
RUN pip install -r requirements.txt