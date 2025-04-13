FROM jupyter/base-notebook:python-3.11

USER root

RUN apt-get update && apt-get install -y make curl

RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        curl -L https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.42/quarto-1.6.42-linux-amd64.deb -o /tmp/quarto.deb; \
    elif [ "$ARCH" = "arm64" ]; then \
        curl -L https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.42/quarto-1.6.42-linux-arm64.deb -o /tmp/quarto.deb; \
    else \
        echo "Unsupported architecture: $ARCH"; exit 1; \
    fi && \
    dpkg -i /tmp/quarto.deb && \
    rm /tmp/quarto.deb


RUN pip install pandas==2.2.3 \
    matplotlib==3.10.1 \
    seaborn==0.13.2 \
    scipy==1.11.3 \
    numpy==1.26.4 \
    scikit-learn==1.3.0 \
    click==8.1.7 \
    requests==2.32.3 \
    pytest==8.3.5 \
    pyarrow==19.0.1

RUN mkdir -p /app && chown -R root:root /app && chmod -R 777 /app && chmod -R 755 /app

WORKDIR /app

COPY . /app/

RUN mkdir -p /root/.jupyter

RUN echo "c.NotebookApp.allow_origin = '*'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_remote_access = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.disable_check_xsrf = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_scripts = True" >> /root/.jupyter/jupyter_notebook_config.py


RUN chown -R root /app

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]