Этот проект предназначен для автоматического расчета границ сцен и относительной громкости звука внутри сцен с последующим вырезанием сцен.
**Использование:**  
> usage: python.exe -m signiscu [-h] [--config CONFIG] [--project PROJECT] {cut,generate,report,build-cut}

Нарезка по project.toml и JSON-кандидату или генерация кандидата (ffmpeg/ffprobe).  

positional arguments:  
{cut,generate,report,build-cut}  

|Аргумент|Описание|
|---|---|
|cut|Разрезать по project.toml и JSON-кандидату (видео — из project.toml и/или JSON)|
|generate|Сгенерировать JSON кандидата по окнам громкости|
|report|HTML-отчёт по JSON-кандидатам в каталоге (*.json)|
|build-cut|Собрать cut JSON из selected_candidates.json (manifest HTML-отчёта)|
|-h, --help|show this help message and exit|
|--config CONFIG|Путь к config.toml (по умолчанию — config.toml рядом с приложением) 
|--project PROJECT|Путь к project.toml при нестандартном расположении (по умолчанию — project.toml рядом с приложением, если есть)  

Флаги --config и --project задаются до подкоманды cut|generate, например: python -m signiscu --project D:/work/project.toml generate  

**Использование подкоманды generate:**  
Подкоманда анализирует видеофайл и записывает выбранные отрезки видео в файла candidate.json на основе настроенных метрик в файле `config.toml`  
>usage: python.exe -m signiscu generate [-h] [-o CANDIDATE_JSON] [--output-dir OUTPUT_DIR] [--list-windows {none,all,selected}] [--debug | --no-debug] [--debug-log DEBUG_LOG] [--csv] [video]  

|Аргумент|Описание|
|---|---|
|video|Входной видеофайл или каталог с файлами (или [input] video в project.toml)|
|-h, --help|show this help message and exit|
|-o, --output CANDIDATE_JSON|JSON кандидата одиночного режима (при каталоге — каталог этого файла; тогда имена candidate_clips_NNNNN.json)|
|--output-dir OUTPUT_DIR|Каталог нарезки в JSON; по умолчанию out или [output] clips_dir|
|--list-windows {none,all,selected}|Вывод окон в stdout: none / all / selected|
|--debug, --no-debug|Отладка: по умолчанию из project.toml; --no-debug отключает явно|
|--debug-log DEBUG_LOG|Файл журнала отладки (переопределяет log_file в TOML)|
|--csv|После generate записать сводный candidate_summary.csv (UTF-8, «;») рядом с JSON кандидатов|

```python.exe -m signiscu generate -o candidate.json video.mp4``` - для одиночоного файла  
```python.exe -m signiscu generate --output-dir candidate-dir video-dir``` - для каталога с видеофайлами  

---

**Использование подкоманды report:**  
Подкоманда герерирует HTML-отчёт на основе существующих файлов candidate.json. В HTML-отчёте можно выбрать отрезки видео и сохранить для дальнейшей работы 
usage: python.exe -m signiscu report [-h] --input-dir INPUT_DIR --output OUTPUT  

|Аргумент|Описание|
|---|---|
|-h, --help|show this help message and exit|
|--input-dir INPUT_DIR|Каталог с JSON кандидатов (любые *.json, название файла произвольное)|
| --output OUTPUT|Путь к выходному HTML-файлу отчёта|

```python.exe -m signiscu report --input-dir candidate-dir --output report.html```  

---

**Использование подкоманды build-cut:**  
Сбор разных выбранных выбранных отрезков видео из HTML-отчёта в единый json-файл для дальнейшей работы  
>usage: python.exe -m signiscu build-cut [-h] --selection SELECTION --input-dir INPUT_DIR --output-dir OUTPUT_DIR

|Аргумент|Описание|
|---|---|
|-h, --help|show this help message and exit|
|--selection SELECTION|Путь к selected_candidates.json, полученных из файла HTML-отчёта|
|--input-dir INPUT_DIR|Каталог, где лежат JSON из подкоманды generate (имена как в candidate_file; обычно те же *.json, что и для report/cut)|
|--output-dir OUTPUT_DIR|Каталог для cut_<stem>.json|

```python.exe -m signiscu build-cut --selection selected_candidates.json --input-dir candidate-dir --output-dir cut-dir```  

---

**Использование подкоманды cut:**  
Программа использует файлы формата json для вырезания отрезков видео, описанных в json-файле путями к видеофайлам и таймкодами  
usage: python.exe -m signiscu cut [-h] [--debug | --no-debug] [--debug-log DEBUG_LOG] [--input-dir INPUT_DIR]

|Аргумент|Описание|
|---|---|
|-h, --help|show this help message and exit|
| --debug, --no-debug|Отладка: по умолчанию из project.toml; --no-debug отключает явно|
| --debug-log DEBUG_LOG|Файл журнала отладки (переопределяет debug.log_file в TOML)|
|--input-dir INPUT_DIR|Каталог с любыми *.json кандидатов. Если задан — переопределяет [input].candidate_file; видео из каждого JSON (input_video)|

```python.exe -m signiscu cut --input-dir cut-dir```  
