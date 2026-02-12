# Универсальный загрузчик моделей

## Описание

`model_load.cpp` - универсальный загрузчик моделей, который:
- Запрашивает путь к модели
- Загружает модель и метаданные
- Автоматически генерирует интерфейс общения на основе метаданных

## Использование

### Запуск с указанием пути:
```bash
.\model_load.exe models/xor
```

### Интерактивный запуск:
```bash
.\model_load.exe
# Программа запросит путь к модели
```

## Формат метаданных

Метаданные сохраняются в файл `{model_path}_metadata.txt` и содержат:
- `model_name` - название модели
- `description` - описание модели
- `input_dim` - размерность входа
- `input_description` - описание формата ввода
- `input_example` - пример ввода
- `input_format` - формат ввода (например, "4 binary values (0 or 1)")
- `output_dim` - размерность выхода
- `output_description` - описание вывода
- `layer_paths` - пути к файлам слоев
- `activations` - список активаций для каждого слоя

## Пример использования

После обучения модели `xor_train.exe` создаются файлы:
- `models/xor_fc1_weight.bin`, `models/xor_fc1_bias.bin`, `models/xor_fc1_meta.txt`
- `models/xor_fc2_weight.bin`, `models/xor_fc2_bias.bin`, `models/xor_fc2_meta.txt`
- `models/xor_metadata.txt` - метаданные модели

Запуск:
```bash
.\model_load.exe models/xor
```

Программа автоматически:
1. Загрузит метаданные из `models/xor_metadata.txt`
2. Покажет приветствие на основе `input_description` и `input_example`
3. Загрузит все слои модели
4. Будет запрашивать ввод в нужном формате
5. Выведет предсказания в нужном формате

## Добавление новой модели

Для добавления новой модели нужно:
1. Создать `{model_name}_train.cpp` для обучения
2. При сохранении модели создать метаданные:
```cpp
mtf::nn::ModelMetadata metadata;
metadata.model_name = "My Model";
metadata.description = "Description";
metadata.input_dim = 4;
metadata.input_description = "Enter 4 values...";
metadata.input_example = "1 2 3 4";
metadata.input_format = "4 numbers";
metadata.output_dim = 1;
metadata.output_description = "Prediction";
metadata.layer_paths = {"models/my_fc1", "models/my_fc2"};
metadata.activations = {"relu", "sigmoid"};
metadata.save("models/my_metadata.txt");
```

3. Использовать `model_load.exe` для загрузки:
```bash
.\model_load.exe models/my
```
