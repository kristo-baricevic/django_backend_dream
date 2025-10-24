source env/bin/activate 
python manage.py makemigrations
python manage.py migrate
python3 manage.py runserver
uvicorn main:app --reload --port 8001
docker compose exec django python manage.py makemigrations
docker compose exec django python manage.py migrate
pip install -r requirements.txt
docker compose logs fastapi --tail=100 -f

uvicorn app.main:app --reload --port 8002
