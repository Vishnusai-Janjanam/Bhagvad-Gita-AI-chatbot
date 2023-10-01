FROM python:3.10

COPY chatbot/code/ /app/

WORKDIR /app

CMD ["pip", "install", "-r", "requirements.txt"]

CMD ["python3", "download.py"]

CMD ["chainlit", "run", "model.py", "-w"]

EXPOSE 8000
