flask flask-orator psycopg2 tensorflow gensim orator-cache redis

pip install -U celery

celery -A sentiment_task worker --loglevel=info