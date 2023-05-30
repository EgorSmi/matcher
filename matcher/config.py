import logging.config
import os

# models
FILES_FOLDER = "files"
TEXT_DEBERTA_MATCHER = os.path.join(FILES_FOLDER, "deberta_v3_matching_common_train")
HEAD_MODEL_FILENAME = os.path.join(FILES_FOLDER, "head_catboost_model.cbm")
PICS_COSINE_SIMILARITY_THRESHOLD = 0.01
PICS_COSINE_BALL_TREE_PATH = os.path.join(FILES_FOLDER, "ball_tree.pkl")
ATTRIBUTE_DEBERTA_MATCHER = os.path.join(FILES_FOLDER, "deberta_v3_matching_attributes_2") # "deberta_v3_matching_common_train"
RUS_TO_ENG_COLORS = os.path.join(FILES_FOLDER, "rus_to_eng_colors.json")
NAME_TO_HEX = os.path.join(FILES_FOLDER, "name_to_hex_dict.json")
NEEDED_ATTRS = os.path.join(FILES_FOLDER, "needed_attrs_test_and_freq500.json")
XLM_ROBERTA_NAME_MATCHER = os.path.join(FILES_FOLDER, "xlm-roberta-names")
RANKED_LIST_PATH = os.path.join(FILES_FOLDER, "ranked_list.pkl")

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
