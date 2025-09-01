source env/bin/activate 
python manage.py makemigrations
python manage.py migrate
python3 manage.py runserver
uvicorn main:app --reload --port 8001