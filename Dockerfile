FROM jupyter/pyspark-notebook:latest

USER root

WORKDIR /home/jovyan/work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER jovyan

EXPOSE 8888

CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]

