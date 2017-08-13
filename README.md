flask flask-orator psycopg2 tensorflow gensim orator-cache cachy

pip install -U celery[redis]

celery -A sentiment_task worker --loglevel=info