# Сохранение и загрузка моделей

## Реализованная функциональность

### 1. Сохранение/загрузка Tensor
- `Tensor::save(filepath)` - сохраняет тензор в бинарный файл
- `Tensor::load(filepath)` - загружает тензор из бинарного файла

### 2. Сохранение/загрузка Dense слоя
- `Dense::save(filepath)` - сохраняет веса и метаданные слоя
- `Dense::load(filepath)` - загружает слой из файлов

Формат сохранения:
- `{filepath}_weight.bin` - веса слоя
- `{filepath}_bias.bin` - смещения (если есть)
- `{filepath}_meta.txt` - метаданные (input_dim, output_dim, use_bias)

### 3. Использование

**Обучение и сохранение:**
```bash
./xor_train.exe
# Модель сохраняется в models/xor_fc1 и models/xor_fc2
```

**Загрузка и использование:**
```bash
./xor_load.exe
# Введите 4 бита (0 или 1) для проверки четности
# Пример: 1 0 1 1
```

## Структура файлов модели

```
models/
├── xor_fc1_weight.bin  # Веса первого слоя (4x32)
├── xor_fc1_bias.bin    # Смещения первого слоя (1x32)
├── xor_fc1_meta.txt    # Метаданные: 4 32 1
├── xor_fc2_weight.bin  # Веса второго слоя (32x1)
├── xor_fc2_bias.bin    # Смещения второго слоя (1x1)
└── xor_fc2_meta.txt    # Метаданные: 32 1 1
```

## Пример использования в коде

```cpp
// Сохранение
mtf::nn::Dense layer(4, 32);
// ... обучение ...
layer.save("models/my_layer");

// Загрузка
auto loaded_layer = mtf::nn::Dense::load("models/my_layer");
```
