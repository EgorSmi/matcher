import json


def preprocess(s: str) -> str:
    import re

    rubbish = re.compile(r"!?<>")
    cleared_s = rubbish.sub("", s)
    cleared_s = re.sub(r"[\([{})\]]", "", cleared_s)
    return cleared_s.lower()


def parse_attributes(s: str) -> str:
    if s is not None:
        d = json.loads(s)
        res_string = []
        for key, values in d.items():
            res_string.append(key)
            res_string.extend(values)
        res_string = " ".join(res_string)
        return preprocess(res_string)
    return ""
