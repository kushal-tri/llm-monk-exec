ARG AWS_REGION

# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# Remove the conda installed symlink for libcurl, which causes an error with curl.
# Fixes the following error:
# curl: /opt/conda/lib/libcurl.so.4: no version information available (required by curl)

# Add Docker's official GPG key:
  RUN apt-get update
  RUN apt-get install ca-certificates curl
  RUN install -m 0755 -d /etc/apt/keyrings
  RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  RUN chmod a+r /etc/apt/keyrings/docker.asc
    
    # Add the repository to Apt sources:
  RUN echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null
  RUN apt-get update
  RUN apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
  
ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
ENV PYTHONPATH /opt/ml/code/
ENV HF_HOME /opt/ml/data/input/.cache

COPY ./requirements.txt /opt/ml/code/requirements.txt
COPY ./setup.py /opt/ml/code/setup.py

RUN pip install -r /opt/ml/code/requirements.txt

COPY . /opt/ml/code/

RUN rm /opt/ml/code/requirements.txt
RUN rm /opt/ml/code/setup.py
CMD service docker start
