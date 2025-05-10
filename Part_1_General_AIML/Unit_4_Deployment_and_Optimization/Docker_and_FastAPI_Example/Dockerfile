FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install libgl1 libglib2.0-0 -y

EXPOSE 8000

COPY src/requirements.txt src/requirements.txt

RUN pip install -r src/requirements.txt

RUN pip cache purge
RUN apt-get clean autoclean
RUN apt-get autoremove --yes
RUN rm -rf /var/lib/{apt,dpkg,cache,log}/

COPY models models
COPY src src

CMD ["fastapi", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]
