FROM python:3.9

COPY online_inference/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

ADD models /app/models
ADD src /app/src
COPY online_inference/app_pred.py /app/app_pred.py
COPY online_inference/app.py /app/app.py

WORKDIR /app/

ENV PATH_TO_MODEL="/app/models/model.pkl"
EXPOSE 8000

CMD python /app/app.py
