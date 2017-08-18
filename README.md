pip install flask flask-orator psycopg2 tensorflow gensim orator-cache redis

pip install -U celery

celery -A task worker --loglevel=info