FROM python:3.11-slim

WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Hugging Face Spaces requires setting up a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the rest of the application files to the container
COPY --chown=user . $HOME/app

# Hugging Face Spaces routes traffic to port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
