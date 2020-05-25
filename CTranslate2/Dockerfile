FROM opennmt/ctranslate2:latest-ubuntu18-gpu 

WORKDIR /app
ENV PYTHONPATH="${PYTHONPATH}:/app" 

COPY . .
COPY pip.conf /root/.pip/pip.conf
RUN pip install -r requirements.txt 

CMD bash


