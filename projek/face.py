import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def save_reference_hist():
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
            hist = calculate_histogram(face_roi)
            np.save('reference_hist.npy', hist)
            print("Histogram referensi telah disimpan.")
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Success", "Histogram referensi telah disimpan.")
            return
        cv2.imshow('Capture Reference Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def face_recognition():
    if not os.path.exists('reference_hist.npy'):
        messagebox.showerror("Error", "Histogram referensi tidak ditemukan! Simpan referensi wajah terlebih dahulu.")
        return
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    recognized = False
    ref_hist = np.load('reference_hist.npy') 
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            hist = calculate_histogram(face_roi)
            similarity = cv2.compareHist(hist, ref_hist, cv2.HISTCMP_CORREL)
            if similarity > 0.8:  
                recognized = True
                cap.release()
                cv2.destroyAllWindows()
                open_atm_interface()
                return
            
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    if not recognized:
        messagebox.showerror("Access Denied", "Face not recognized!")

def open_atm_interface():
    atm_window = tk.Tk()
    atm_window.title("ATM Interface")
    tk.Label(atm_window, text="Welcome to ATM", font=("Arial", 20)).pack(pady=20)
    atm_window.mainloop()

root = tk.Tk()
root.title("Face Recognition ATM")
tk.Label(root, text="PILIH UNTUK AUTENTIKASI", font=("Arial", 14)).pack(pady=20)
start_btn = tk.Button(root, text="Mulai", command=face_recognition)
start_btn.pack(pady=5)
save_btn = tk.Button(root, text="Daftarkan Wajah", command=save_reference_hist)
save_btn.pack(pady=5)


root.mainloop()
