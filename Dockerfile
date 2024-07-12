FROM python:3.11-bookworm
ENV PYTHONUNBUFFERED=1

# create code directory
RUN mkdir /code
WORKDIR /code

# install python requirements
RUN pip install --upgrade pip
# RUN apt-get update && apt-get install -y libgl1-mesa-glx

# copy just requirements and install before rest of code to avoid having to
# reinstall packages during build every time code changes
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# copy code files
COPY . /code/

CMD ["python", "app.py"]
EXPOSE 5000