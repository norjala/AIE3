FROM python:3.11
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY . .
RUN chown -R user:user .
RUN pip install -r requirements.txt
USER user
CMD ["chainlit", "run", "app.py", "--port", "7860"]