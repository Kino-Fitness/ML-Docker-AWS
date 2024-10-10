FROM python:3.11-bookworm
ENV PYTHONUNBUFFERED=1

# create code directory
RUN mkdir /code
WORKDIR /code

# install python requirements
RUN pip install --upgrade pip

# Install AWS CLI
RUN apt-get update && \
    apt-get install -y awscli && \
    apt-get clean

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# copy just requirements and install before rest of code to avoid having to
# reinstall packages during build every time code changes
COPY requirements/ /code/requirements/
COPY requirements/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/requirements.txt

# copy code files
COPY . /code/

CMD ["python", "app.py"]
EXPOSE 5000