FROM continuumio/miniconda3:latest
RUN conda install python=3.8
ENV APP_HOME /usr/app
COPY ./requirements.txt $APP_HOME/requirements.txt
RUN pip3 install -r $APP_HOME/requirements.txt
COPY ./model/XGBC_model.pkl $APP_HOME/model/XGBC_model.pkl
COPY ./docker/ $APP_HOME
WORKDIR $APP_HOME
EXPOSE 5000:5000
CMD python app.py
