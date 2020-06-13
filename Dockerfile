# using this since pytorch images still use python 3.7
FROM anibali/pytorch:1.5.0-cuda10.2
USER root
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /app
VOLUME /app
VOLUME /app/data
ENTRYPOINT ["python3"]
