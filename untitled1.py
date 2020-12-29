# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:26:40 2020

@author: Gurkirat
"""
import tkinter
from tkinter import Frame,Label,Button
from age_emotion import fun,fun1
from record import ui
from PIL import Image, ImageTk

sroot = Tk()
sroot.title('Emotion, Age and Gender Detection')
sroot.minsize(height=500,width=1140)
sroot.configure(bg='white')
facial="C:/Users/Gurkirat/Desktop/New Folder/mainpage.png"
img0 = ImageTk.PhotoImage(Image.open(facial))
panel = Button(sroot,image=img0)
panel.image = img0
panel.place(x=0,y=0)

#chilanka
#Label(sroot,text="Emotion, Age and Gender Detection",font='Timesnewroman 40 ',bg='white',fg='black').place(x=535,y=10)

def main():

    window=tkinter.Tk()
    
    window.title('Emotion, Age and Gender Detection')
    window.geometry('1000x700')


    window = Frame( window)
    window.place(relx=0.02, rely=0.03, relheight=0.94, relwidth=0.96)
    window.configure(borderwidth="2")
    window.configure(background="#d9d9d9")
    window.configure(highlightbackground="#d9d9d9")
    window.configure(highlightcolor="black")
    window.configure(width=925)
    
    window.configure(background="#d9d9d9")
    window.configure(highlightbackground="#d9d9d9")
    window.configure(highlightcolor="black")
    Label(window,text = "\t\t\t Emotion, Age and Gender Detection \t\t\t\t",font=('Calibri',26,'bold'), bg="#d9d9d9",height=3).pack()
    b0=Button(window,padx=16,pady=4,bd=4,fg="white",font=('arial',16,'bold'),width=21,height=2,bg="gray25",text='Image Recognition',command=fun1)
    b0.place(x=70,y=150)
    b1=Button(window,padx=16,pady=4,bd=4,fg="white",font=('arial',16,'bold'),width=21,height=2,bg="gray25",text='Face Recognition',command=fun)
    b1.place(x=70,y=327)
    b2=Button(window,padx=16,pady=4,bd=4,fg="white",font=('arial',16,'bold'),width=21,height=2,bg="gray25",text='Voice Recognition',command=ui)
    b2.place(x=70,y=500)

    
    facial="C:/Users/Gurkirat/Desktop/New Folder/istockfae.jpg"
    img0 = ImageTk.PhotoImage(Image.open(facial))
    panel = Button(window, image = img0, command=fun1)
    panel.image = img0
    panel.place(x=600,y=150)
    face="C:/Users/Gurkirat/Desktop/New Folder/face.jpg"
    img1 = ImageTk.PhotoImage(Image.open(face))
    panel = Button(window, image = img1,command=fun)
    panel.image = img1
    panel.place(x=600,y=327)
    voice="C:/Users/Gurkirat/Desktop/New Folder/voice.jpg"
    img2 = ImageTk.PhotoImage(Image.open(voice))
    panel2 = Button(window, image = img2,command=ui)
    panel2.image = img2
    panel2.place(x=600,y=500)
    window.mainloop()

def call_mainroot():
	sroot.destroy()
	main()
sroot.after(3500,call_mainroot)
    
mainloop()