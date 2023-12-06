# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import time
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from decouple import config

IMG_FILE_DEF          = os.path.abspath("../test/img3.jpg")
FILE_TMP              = "tmp.jpg"
# Load environment variables from .env
SERVER_NAME           = config("SERVER_NAME")
SERVER_URL            = config("SERVER_URL")
IS_GCLOUD_DEP         = config("IS_GCLOUD_DEP", cast=bool)
IS_GRPC_USE           = config("IS_GRPC_USE", cast=bool)
IS_DOCKER_USE         = config("IS_DOCKER_USE", cast=bool)

base_url = SERVER_URL if IS_GCLOUD_DEP\
  else SERVER_NAME if IS_DOCKER_USE else "localhost"
port_suffix = "8001" if IS_GRPC_USE else "8000"
url = f"{base_url}:{port_suffix}"
print(f"Connecting to {url} ...")

client = grpcclient.InferenceServerClient(url) if IS_GRPC_USE else httpclient.InferenceServerClient(url)


def inference(img=None):
  """Inference function for Text Detection

  Parameters
  ----------
  img : Image, optional
      Input image, by default None

  Returns
  -------
  text : string
      Recognized text
  """

  # Read image
  if img is not None:
    # Store image to temporary file
    img.save(FILE_TMP)
    # Keep checking if temporary file exists in 5 seconds.
    start_time = time.time()
    while not os.path.exists(FILE_TMP):
      time.sleep(1)
      # Raise error if file not found in 5 seconds
      if time.time() - start_time > 5:
        raise Exception("Image not found")

    # Read image from temporary file
    image_data = np.fromfile(FILE_TMP, dtype="uint8")
    # Delete temporary file    
    os.remove(FILE_TMP)
  else:
    image_data = np.fromfile(IMG_FILE_DEF, dtype="uint8")

  print(f"Type of image {type(image_data)}")
  print(f"Image shape {image_data.shape}")
  print(f"Image {image_data}")
  image_data = np.expand_dims(image_data, axis=0)

  input_tensors = [grpcclient.InferInput("input_image", image_data.shape, "UINT8")] if IS_GRPC_USE\
    else [httpclient.InferInput("input_image", image_data.shape, "UINT8")]
  input_tensors[0].set_data_from_numpy(image_data)

  # Check inference time consumption
  start_time = time.time()
  results = client.infer(model_name="ensemble_model", inputs=input_tensors)
  end_time = time.time()

  # Display inference time consumption in seconds (get 3 decimal places)
  print("Inference time: {:.3f}s".format(end_time - start_time))
  output_data = results.as_numpy("recognized_text").astype(str)
  print(output_data)
  return output_data

if __name__ == "__main__":
  inference()