FROM jupyter/base-notebook:python-3.11


RUN pip install pandas==2.2.3 \
    matplotlib==3.10.1 \
    seaborn==0.13.2 \
    scipy==1.11.3 \
    numpy==1.26.4 \
    scikit-learn==1.3.0

COPY . /app/

WORKDIR /app

RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py

USER root
RUN chown -R root /app

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]