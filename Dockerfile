FROM python:3.7

# install build utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get -y upgrade

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir app
WORKDIR /app
COPY . /app

CMD ["python", "app.py"]