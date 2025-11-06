source env/bin/activate 
brew services start postgresql@14
python manage.py makemigrations
python manage.py migrate
python3 manage.py runserver
uvicorn main:app --reload --port 8001
docker compose exec django python manage.py makemigrations
docker compose exec django python manage.py migrate
pip install -r requirements.txt
docker compose logs fastapi --tail=100 -f
docker compose up --build
uvicorn app.main:app --reload --port 8002\
python manage.py shell

