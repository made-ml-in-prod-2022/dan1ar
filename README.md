# ML Prod HW#2
Команды исполяются из корневой директории проекта

Создание docker image:
- Локально
    ~~~
    docker build -t dan1ar/ml_prod_hw2 -f online_inference/Dockerfile .
    ~~~

- Через docker hub
    ~~~
    docker pull dan1ar/ml_prod_hw2
    ~~~

Запуск контейнера с сервером
~~~
docker run --rm -it -p 8000:8000 dan1ar/ml_prod_hw2
~~~

Сделать запрос на сервер
~~~
cd online_inference
python3 make_request.py
# Если сервер запущен в контейнере, то
python3 make_request.py -i localhost
~~~

Датасет: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uc

**Основная часть**

- [x] Оберните inference вашей модели в rest сервис на FastAPI, должен быть endpoint /predict (3 балла)
- [x] Напишите endpoint /health (1 балл), должен возращать 200, если ваша модель готова к работе (такой чек особенно актуален если делаете доп задание про скачивание из хранилища) 
- [x] Напишите unit тест для /predict  (3 балла) (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/)

- [x] Напишите скрипт, который будет делать запросы к вашему сервису -- 2 балла

- [x] Напишите dockerfile, соберите на его основе образ и запустите локально контейнер (docker build, docker run), внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки (4 балл)

- [x] опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (+2 балла)

- [x] напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель (1 балл)
   Убедитесь, что вы можете протыкать его скриптом из пункта 3

- [x] проведите самооценку (распишите в реквесте какие пункты выполнили и на сколько баллов, укажите сумму баллов) -- 1 балл

