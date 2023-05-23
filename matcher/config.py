import logging.config

# models
TEXT_DEBERTA_MATCHER = "deberta_v3_matching_common_train"
HEAD_MODEL_FILENAME = "head_catboost_model.cbm"
PICS_COSINE_SIMILARITY_THRESHOLD = 0.01

# columns
SCORE_COLUMN = "target"


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
            "level": "INFO",
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
