FROM python:3.12    
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH 
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --user langchain-qdrant
RUN pip install -r requirements.txt
RUN pip list

# Create the directory and set permissions
USER root
RUN mkdir -p /home/user/app/data/vectorstore && \
    chown -R user:user /home/user/app/data && \
    chmod -R 777 /home/user/app/data
USER user

CMD ["chainlit", "run", "app.py", "--port", "7860"]
