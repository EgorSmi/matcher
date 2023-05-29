# Uranus Matcher
Мы предоставляем файл solution.tar.gz с окружением solution-env.tar.gz и скриптом pipeline.py

Для запуска решения вам потребуется скачать архив solution.tar.gz, затем следовать инструкциям описанным ниже.

## Как запускать решение

1) Создаем директорию, в которую распакуем решение: mkdir solution
2) Разархивируем: tar -xzf solution.tar.gz -C solution
3) Зайдем в директорию: cd solution 
4) Создаем директорию под conda окружение: mkdir solution-env 
5) Разархивируем: tar -xzf solution-env.tar.gz -C solution-env
6) Активируем окружение: source solution-env/bin/activate
7) Запускаем скрипт с решением указывая пути до тестовых файлов: python3 pipeline.py main "<path-to-test-pairs.parquet>" "<path-to-test-data.parquet>"


На выходе вы получите файл submission.csv со скорами.