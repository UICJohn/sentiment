pip3 install flask flask-orator psycopg2 tensorflow gensim orator-cache redis

pip3 install -U celery

celery -A task worker --loglevel=info