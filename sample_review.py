# Sample file with review findings

import subprocess
import pickle


def unsafe_eval(user_input):
    result = eval(user_input)  # High: eval executes dynamic code
    return result


def run_command(cmd):
    subprocess.run(cmd, shell=True)  # High: shell=True is unsafe


def load_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)  # High: pickle.load can execute arbitrary code


def process():
    try:
        something()
    except:  # Medium: bare except catches everything
        pass


API_KEY = "sk-1234567890abcdef"  # High: hardcoded credential
