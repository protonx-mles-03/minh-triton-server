# Base image and infos
FROM python:3.8
LABEL maintainer="nhatminh.cdt@gmail.com" \
      description="Application for text detection model" \
      version="1.0"

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}
# To make app accessible on local network
ENV GRADIO_SERVER_NAME=0.0.0.0

# Prepare environment
WORKDIR /app
ADD requirements.txt ./requirements.txt
# Add all python files to the container
COPY client.py app.py .env ./

# Install requirements
RUN pip install -r requirements.txt

CMD [ "python", "-u", "app.py" ]

