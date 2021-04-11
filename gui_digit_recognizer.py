# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 20:58:24 2021

@author: Harsh
"""
from keras.models import load_model
import tkinter as tk
from tkinter import *
import win32gui
from PIL import ImageGrab, Image
import numpy as np

model = load_model('model_hdlenet5.h5')

def predict_digit(img):
    #resize image to 28X28
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #prediction 
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class Abc(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        self.x = 0
        self.y = 0
        
        #initializing elements
        
        self.canvas = tk.Canvas(self,bd = 20, width = 300, height = 300, bg = "white", cursor = "cross")
        self.label = tk.Label(self,text = "Draw a single digit.", bg = "black", font =('Comic Sans MS',15), width = 22, fg = "white", anchor = "w")
        
        self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting)   
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
       
        
        self.canvas.grid(row = 1, column = 0, pady = 2, sticky = W)
        self.label.grid(row = 0,column = 0, pady = 2, padx = 2, columnspan = 3)
        self.classify_btn.grid(row = 2, column = 0)
        self.button_clear.grid(row = 3, column = 0)
        
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        
    def clear_all(self):
        self.canvas.delete("all")
        
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        a,b,c,d = rect
        rect=(a+4,b+4,c-4,d-4)
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)

        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
       
        
abc = Abc()
mainloop()