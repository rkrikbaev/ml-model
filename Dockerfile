FROM python:3.7

# install build utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get -y upgrade

# Install dependencies
RUN pip install flask && \
    pip install tensorflow && \
    pip install pandas && \
    pip install numpy && \
    pip install scipy && \
    pip install pyyaml

RUN mkdir app
WORKDIR /app
COPY . /app

CMD ["python", "app.py"]