# Uranus Matcher

## Как запускать решение

Мв предоставляем файл solution-env.tar.gz, который нужно распаковать и активировать

## Последовательность действий
1) mkdir solution-env
2) tar -xzf solution-env.tar.gz -C solution-env
3) source solution-env/bin/activate
4) conda-unpack
5) python matcher/pipeline.py main "test_pairs.parquet" "test_data.parquet"


На выходе вы получите файл submission.csv со скорами.