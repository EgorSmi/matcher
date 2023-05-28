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
            res_string.append(values)
        res_string = " ".join(res_string)
        return preprocess(res_string)
    return ""


def append_color_to_attrs(attr_dict, color_parsed: str) -> str:
    if attr_dict is not None:
        attr_dict = json.loads(attr_dict)
        if color_parsed is not None:
            attr_dict["Цвет"] = color_parsed
        return json.dumps(dict(sorted(attr_dict.items(), key=lambda item: item[0])),
                              ensure_ascii=False)
    return None
