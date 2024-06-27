FROM python:3.9

RUN pip install --upgrade pip

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY ./data /home/user/app/data  

# Copy only requirements.txt first to leverage Docker cache
COPY --chown=user requirements.txt $HOME/app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . $HOME/app

# Run the application
CMD ["chainlit", "run", "app.py", "--port", "7860"]
