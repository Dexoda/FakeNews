import os
import yaml
import logging

# Настройка логгера
logger = logging.getLogger(__name__)

# Путь к файлу конфигурации по умолчанию
DEFAULT_CONFIG_PATH = "/app/config.yml"

def load_config(config_path=None):
    """
    Загружает конфигурацию из YAML-файла.
    
    Args:
        config_path (str, optional): Путь к файлу конфигурации. 
                                    Если не указан, используется значение из
                                    переменной окружения CONFIG_PATH или путь по умолчанию.
    
    Returns:
        dict: Словарь с конфигурацией
    
    Raises:
        FileNotFoundError: Если файл конфигурации не найден
        yaml.YAMLError: Если файл конфигурации содержит недопустимый YAML
    """
    # Определение пути к файлу конфигурации
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    
    logger.info(f"Загрузка конфигурации из {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Проверка наличия необходимых разделов конфигурации
        if "analysis" not in config:
            config["analysis"] = {}
        
        if "semantic" not in config["analysis"]:
            config["analysis"]["semantic"] = {
                # Значения по умолчанию для модели spaCy
                "model_name": "ru_core_news_md"
            }
            
        logger.info("Конфигурация успешно загружена")
        return config
        
    except FileNotFoundError as e:
        logger.error(f"Файл конфигурации не найден: {config_path}")
        # Создаем базовую конфигурацию по умолчанию
        default_config = {
            "analysis": {
                "semantic": {
                    "model_name": "ru_core_news_md"
                }
            }
        }
        logger.warning("Используется конфигурация по умолчанию")
        return default_config
        
    except yaml.YAMLError as e:
        logger.error(f"Ошибка разбора YAML-файла: {e}")
        raise
