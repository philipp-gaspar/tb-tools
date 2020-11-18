# use the official tensorflow GPU as a parent image
FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer = "philipp.gaspar@gmail.com"

ARG USERNAME
ARG UID
ARG GID

RUN groupadd -g $GID -o $USERNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $USERNAME

# --------------------- #
#    PYTHON PACKAGES    #
# ===================== #
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt && requirements.txt