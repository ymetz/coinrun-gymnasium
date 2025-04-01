FROM python:3.8-slim
RUN apt-get update
RUN apt-get install --yes build-essential qt5-default pkg-config
ADD . coinrun
RUN pip install -e coinrun
# this has the side-effect of building the coinrun env
RUN python -c 'import coinrun'