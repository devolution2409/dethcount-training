FROM tensorflow/tensorflow:latest-py3

RUN pip install --upgrade pip && pip install tensorflow_datasets && pip install --upgrade tensorflow keras numpy pandas sklearn pillow


#&& apt-get upgrade -y && apt-get install -y git


WORKDIR /source/

COPY .entrypoint.sh /.entrypoint.sh
RUN ["chmod", "+x","/.entrypoint.sh"]
ENTRYPOINT [ "sh","/.entrypoint.sh" ]

RUN bash