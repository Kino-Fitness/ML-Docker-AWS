FROM python:3.11-bookworm
ENV PYTHONUNBUFFERED=1

# Create code directory
RUN mkdir /code
WORKDIR /code

# Install AWS CLI and libgl1-mesa-glx in a single step and clean up after installation
RUN apt-get update && \
    apt-get install -y awscli libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy just requirements and install before rest of code to avoid having to
# reinstall packages during build every time code changes
COPY requirements/ /code/requirements/
COPY requirements/requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/requirements.txt

# Copy code files
COPY . /code/

# Initialize app and run Gunicorn server

CMD python initalize.py && app.py
EXPOSE 5000