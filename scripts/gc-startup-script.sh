#! /bin/bash
# Startup script for instance
TRITON_CONTAINER_DIR = "asia.gcr.io/mles-class-01/text-detection-triton:latest"
MODEL_REPOSITORY_DIR = "gs://mle-class-project/model_repository"
cd /home
sudo snap install docker
sudo snap start docker
gcloud auth configure-docker
gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin https://asia.gcr.io
# Copy model repository from Google Cloud Storage
gsutil cp -r $MODEL_REPOSITORY_DIR .
# Run Triton server Docker container from Google Container Registry
sudo docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models $TRITON_CONTAINER_DIR tritonserver --model-repository=/models
