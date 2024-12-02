import torch
import os
import json

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_json(json_path):
    with open(json_path, 'r', encoding="UTF-8") as file:
        data = json.load(file)
    file.close()
    return data

def setting2json(obj):
    """재귀적으로 튜플을 리스트로 변환"""
    if isinstance(obj, dict):
        return {key: setting2json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [setting2json(element) for element in obj]
    elif isinstance(obj, tuple):  # 튜플을 리스트로 변환
        return list(obj)
    else:
        return obj