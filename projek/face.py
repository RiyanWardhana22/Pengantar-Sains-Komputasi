import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt

def perhitungan_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def plot_histogram(hist, title="Histogram"):
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.plot(hist, color='red')
    plt.xlim([0, 256])
    plt.show()

def simpan_daftar_histogram():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            hist = perhitungan_histogram(face_roi)
            np.save('projek/reference_hist.npy', hist)
            print("Histogram referensi telah disimpan.")
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Success", "Histogram referensi telah disimpan.")
            plot_histogram(hist, title="Reference Face Histogram")
            return
        cv2.imshow('Capture Reference Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def face_recognition():
    if not os.path.exists('projek/reference_hist.npy'):
        messagebox.showerror("Error", "Histogram referensi tidak ditemukan! Simpan referensi wajah terlebih dahulu.")
        return
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    recognized = False
    ref_hist = np.load('projek/reference_hist.npy') 
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            hist = perhitungan_histogram(face_roi)
            similarity = cv2.compareHist(hist, ref_hist, cv2.HISTCMP_CORREL)
            if similarity > 0.8:  
                recognized = True
                cap.release()
                cv2.destroyAllWindows()
                plot_histogram(hist, title="Recognized Face Histogram")
                atm()
                return
            else:
                plot_histogram(hist, title="Unrecognized Face Histogram")
            
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    if not recognized:
        messagebox.showerror("Access Denied", "Face not recognized!")

def atm():
    atm_window = ThemedTk(theme="arc")
    atm_window.title("ATM Interface")
    atm_window.geometry("400x300")
    tk.Label(atm_window, text="Welcome to ATM", font=("Arial", 20)).pack(pady=20)
    atm_window.mainloop()

root = ThemedTk(theme="arc")
root.title("Face Recognition ATM")
root.geometry("400x300")

image = Image.open("projek/atm.png")  
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(root, image=photo)
image_label.pack(pady=10)

title_label = tk.Label(root, text="PILIH UNTUK AUTENTIKASI", font=("Arial", 14))
title_label.pack(pady=10)

style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)

start_btn = ttk.Button(root, text="Mulai", command=face_recognition)
start_btn.pack(pady=5)

save_btn = ttk.Button(root, text="Daftarkan Wajah", command=simpan_daftar_histogram)
save_btn.pack(pady=5)

root.mainloop()