import logging.config


TEXT_DEBERTA_MATCHER = "deberta_v3_matching_common_train"



# log
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(levelname)-8s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(message)s",
        },
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "matcher": {
            "level": "DEBUG",
            "handlers": [
                "default",
            ],
        }
    },
    "root": {
        "level": "INFO",
        "handlers": [
            "default",
        ],
    },
}
logging.config.dictConfig(LOGGING_CONFIG)