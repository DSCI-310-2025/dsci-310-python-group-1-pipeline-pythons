FROM jupyter/base-notebook:python-3.11

USER root

RUN pip install pandas==2.2.3 \
    matplotlib==3.10.1 \
    seaborn==0.13.2 \
    scipy==1.11.3 \
    numpy==1.26.4 \
    scikit-learn==1.3.0\
    click==8.1.7 \
    requests==2.31.0

RUN mkdir -p /app && chown -R root:root /app && chmod -R 777 /app

WORKDIR /app

COPY . /app/

RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py


RUN chown -R root /app

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]