#---DOCKER--
# Notice: Assume that the gcloud project has ID "mles-class-01"
# To build docker image
docker build ./ -f Dockerfile -t asia.gcr.io/mles-class-01/text-detection-triton:latest
docker build ./ -f Dockerfile -t asia.gcr.io/mles-class-01/text-detection-app:latest

# To run server docker image locally
docker run --gpus=all -it --shm-size=1g --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models asia.gcr.io/mles-class-01/text-detection-triton:latest
tritonserver --model-repository=/models
# To run client docker image locally
docker run --name text-detection-app --rm -p7860:7860 asia.gcr.io/mles-class-01/text-detection-app:latest

# To push docker image to Google Cloud Registry (GCR)
# Notice: gcloud project authentification is required, see gcloud-script.sh
docker push asia.gcr.io/mles-class-01/text-detection-triton:latest
docker push asia.gcr.io/mles-class-01/text-detection-app:latest
