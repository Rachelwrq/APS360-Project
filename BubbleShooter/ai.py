import puzzlebobble as bb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
import pynput
from pynput.keyboard import Key, Controller


def shoot_rand():
    direction = None
    keyboard = Controller()
    for i in range(30):
        dir_int = randint(0,1)
        if dir_int == 0:
            event.key = LEFT
            print ("left")
        else:
            event.key = RIGHT
    #launchBubble = True
    return "cock"


def feed_input():
    print("cock")
'''
class baseline_ANN(nn.Module):
    def __init__(self):
        super(baseline_ANN).__init__()
'''

if __name__ == "__main__":
    bb.main()
    i = 0
    while i < 999:
        shoot_rand()
        i += 1