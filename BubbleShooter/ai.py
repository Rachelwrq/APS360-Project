import puzzlebobble as bb
from puzzlebobble import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
import pynput
from pynput.keyboard import Key, Controller
import math, pygame, sys, os, copy, time, random
import pygame.gfxdraw
from pygame.locals import *

FPS          = 120
WINDOWWIDTH  = 640
WINDOWHEIGHT = 480
TEXTHEIGHT   = 20
BUBBLERADIUS = 20
BUBBLEWIDTH  = BUBBLERADIUS * 2
BUBBLELAYERS = 5
BUBBLEYADJUST = 5
STARTX = WINDOWWIDTH / 2
STARTY = WINDOWHEIGHT - 27
ARRAYWIDTH = 16
ARRAYHEIGHT = 14
LISTSIZE = ARRAYWIDTH * ARRAYHEIGHT


RIGHT = 'right'
LEFT  = 'left'
BLANK = '.'

## COLORS ##

#            R    G    B
GRAY     = (100, 100, 100)
NAVYBLUE = ( 60,  60, 100)
WHITE    = (255, 255, 255)
RED      = (255,   0,   0)
GREEN    = (  0, 255,   0)
BLUE     = (  0,   0, 255)
YELLOW   = (255, 255,   0)
ORANGE   = (255, 128,   0)
PURPLE   = (255,   0, 255)
CYAN     = (  0, 255, 255)
BLACK    = (  0,   0,   0)
COMBLUE  = (233, 232, 255)

BGCOLOR    = WHITE
COLORLIST = [RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, CYAN]

def init_game():
    gameColorList = copy.deepcopy(COLORLIST)
    direction = None
    launchBubble = False
    newBubble = None
    
    
    
    arrow = Arrow()
    bubbleArray = makeBlankBoard()
    setBubbles(bubbleArray, gameColorList)
    
    nextBubble = Bubble(gameColorList[0])
    nextBubble.rect.right = WINDOWWIDTH - 5
    nextBubble.rect.bottom = WINDOWHEIGHT - 5

    score = Score()

def runGame1(input_event, newBubble, nextBubble, arrow, bubbleArray, score, direction):
    
    if input_event == "left":
        direction = LEFT
    elif input_event == "right":
        direction = RIGHT
    elif input_event == "shoot":
        direction = RIGHT
        launchBubble = True

    # for event in input_event:            
    #     if event.type == KEYDOWN:
    #         if (event.key == K_LEFT):
    #             direction = LEFT
    #         elif (event.key == K_RIGHT):
    #             direction = RIGHT
                
    #     elif event.type == KEYUP:
    #         direction = None
    #         if event.key == K_SPACE:
    #             launchBubble = True
    #         elif event.key == K_ESCAPE:
    #             terminate()

    if launchBubble == True:
        if newBubble == None:
            newBubble = Bubble(nextBubble.color)
            newBubble.angle = arrow.angle
            

        newBubble.update()
        newBubble.draw()
        
        
        if newBubble.rect.right >= WINDOWWIDTH - 5:
            newBubble.angle = 180 - newBubble.angle
        elif newBubble.rect.left <= 5:
            newBubble.angle = 180 - newBubble.angle


        launchBubble, newBubble, score = stopBubble(bubbleArray, newBubble, launchBubble, score)

        finalBubbleList = []
        for row in range(len(bubbleArray)):
            for column in range(len(bubbleArray[0])):
                if bubbleArray[row][column] != BLANK:
                    finalBubbleList.append(bubbleArray[row][column])
                    # if bubbleArray[row][column].rect.bottom > (WINDOWHEIGHT - arrow.rect.height - 10):
                    #     return score.total, finalBubbleList, 'lose'

        
        
        # if len(finalBubbleList) < 1:
        #     return score.total, finalBubbleList, 'win'
                                    
                    
        
        gameColorList = updateColorList(bubbleArray)
        random.shuffle(gameColorList)
        
                
                        
        if launchBubble == False:
            
            nextBubble = Bubble(gameColorList[0])
            nextBubble.rect.right = WINDOWWIDTH - 5
            nextBubble.rect.bottom = WINDOWHEIGHT - 5
    #return score, finalBubbleList, 'nothing'
    
                        
    nextBubble.draw()
    if launchBubble == True:
        coverNextBubble()
    
    arrow.update(direction)
    arrow.draw()


    
    setArrayPos(bubbleArray)
    drawBubbleArray(bubbleArray)

    #score.draw()
    #while True:
    pygame.display.update()
    
    #FPSCLOCK.tick(FPS)
    return score, finalBubbleList, 'nothing'


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
    return

class baseline_ANN(nn.Module):
    def __init__(self):
        super(baseline_ANN, self).__init__()
        self.name = "baseline"
        self.layer = nn.Linear(LISTSIZE, 3)
    def forward(self, arr):
        activation = self.layer(self)
        output = F.relu(activation)
        return output

def main():
    gameColorList = copy.deepcopy(COLORLIST)
    direction = None
    launchBubble = False
    newBubble = None
    
    
    
    arrow = Arrow()
    bubbleArray = makeBlankBoard()
    setBubbles(bubbleArray, gameColorList)
    
    nextBubble = Bubble(gameColorList[0])
    nextBubble.rect.right = WINDOWWIDTH - 5
    nextBubble.rect.bottom = WINDOWHEIGHT - 5

    score = Score()
    #while True:
    #runGame1("shoot", newBubble, nextBubble, arrow, bubbleArray, score, direction)
    while True:
        score, bubble_arr, winorlose = runGame1("shoot", newBubble, nextBubble, arrow, bubbleArray, score, direction)
    #winorlose = "win"
    endScreen(score, winorlose)


if __name__ == "__main__":
    init()
    main()


