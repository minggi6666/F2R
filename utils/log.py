"""Small terminal-color helpers used by training logs."""

import logging
import os

class Color:
    Black = 0
    Red = 1
    Green = 2
    Yellow = 3
    Blue = 4
    Magenta = 5
    Cyan = 6
    White = 7

class Mode:
    Foreground = 30
    Background = 40
    ForegroundBright = 90
    BackgroundBright = 100

def tcolor(txt, c, m=Mode.Foreground):
    """
    Commonly used color escape functions.
    Example: print(tcolor("Learning rate: 0.001", c=Color.Magenta))
    """
    return '\033[{}m'.format(m + c) + txt + '\033[0m'

def gradient_num_color(value, v_min=38.6, v_max=39.5, ascending=True):
    """
    24-bit gradient color numeric escape function for terminal output.
    """
    c_value = min(v_max, max(v_min, value))
    color = (c_value - v_min) / (v_max - v_min) * 255
    if ascending:
        color = 255 - color
    r = int(color)
    g = int(255 - color)
    return '\033[38;2;{};{};{}m'.format(r, g, 0) + "{:.4f}".format(value) + '\033[0m'

def gen_log(model_path):
    """
    Generate a logger that outputs to both terminal and a log.txt file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = os.path.join(model_path, 'log.txt')
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger