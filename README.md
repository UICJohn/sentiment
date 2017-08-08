flask flask-orator psycopg2 tensorflow gensim 

pip install -U celery[redis]

celery -A sentiment_task worker --loglevel=info