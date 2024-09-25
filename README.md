## Сегментация 
Модели: DeepLab,FPN,Unet

### Настройка доступа к серверу

Указать username и путь к private_key в файле [Makefile](Makefile)

### Датасет

Скачать датасет (он окажется в папке dataset_coco):

```bash
make download_dataset
```

### Подготовка окружения

1. Создание и активация окружения
    ```bash
    python3 -m venv venv
    . venv/bin/activate 
   или
    . venv/Scripts/activate 
    ```

2. Установка библиотек
   ```
    make install
   ```
   
3. Запуск линтеров
   ```
   make lint
   ``` 

4. Логи в ClearML

5. Настраиваем [configs](configs) под себя.


### Обучение

Запуск тренировки:

```bash
make train
```

### Тест

Запуск тестирования(можно настроить конфиг [eval.yaml](configs/eval.yaml), например указать определенный checkpoint) :

```bash
make test
```

### Удаление логов

```bash
make clean-logs
```

### Инференс

Посмотреть результаты работы обученной сети можно в [тетрадке](notebooks/inference.ipynb).

