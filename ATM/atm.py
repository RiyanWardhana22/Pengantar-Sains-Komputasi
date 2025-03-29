import customtkinter as ctk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import pickle
from tkinter import messagebox
import matplotlib.pyplot as plt

class FaceLoginSystem(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ATM Face Login System")
        self.geometry("1000x650")
        self.minsize(900, 600)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue") 

        self.nama = ctk.StringVar()
        self.nim = ctk.StringVar()
        
        self.cap = None
        self.video_label = None
        self.after_id = None
        self.create_widgets()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if os.path.exists('trainer/trainer.yml'):
            self.recognizer.read('trainer/trainer.yml')
    
    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        main_frame = ctk.CTkFrame(self, corner_radius=15)
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        left_frame = ctk.CTkFrame(main_frame, width=500, corner_radius=10)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(left_frame, text="Kamera", width=480, height=360)
        self.video_label.pack(pady=20, padx=10)
        
        self.status_label = ctk.CTkLabel(left_frame, text="Ready", font=("Arial", 12), text_color="gray")
        self.status_label.pack(pady=5)
        
        right_frame = ctk.CTkFrame(main_frame, width=400, corner_radius=10)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        title_label = ctk.CTkLabel(right_frame, text="SISTEM AUTENTIKASI ATM", 
                                 font=("Arial", 20, "bold"))
        title_label.pack(pady=(20, 10))
        
        subtitle_label = ctk.CTkLabel(right_frame, text="Verifikasi Sebelum Login", 
                                    font=("Arial", 12), text_color="gray")
        subtitle_label.pack(pady=(0, 20))
        
        input_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        input_frame.pack(pady=10, padx=10, fill="x")
        
        ctk.CTkLabel(input_frame, text="NAMA:", font=("Arial", 12)).pack(anchor="w", pady=(5, 0))
        name_entry = ctk.CTkEntry(input_frame, textvariable=self.nama, 
                                 placeholder_text="Masukkan Nama Anda", 
                                 height=40, corner_radius=8,
                                 font=("Arial", 12))
        name_entry.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(input_frame, text="NIM:", font=("Arial", 12)).pack(anchor="w", pady=(5, 0))
        nim_entry = ctk.CTkEntry(input_frame, textvariable=self.nim, 
                               placeholder_text="Masukkan NIM", 
                               height=40, corner_radius=8,
                               font=("Arial", 12))
        nim_entry.pack(fill="x", pady=(0, 15))
        
        button_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        button_frame.pack(pady=10, fill="x")
        
        capture_btn = ctk.CTkButton(button_frame, text="Tambah Data Wajah", 
                                   command=self.take_face_samples,
                                   height=40, corner_radius=8,
                                   fg_color="#4e8cff", hover_color="#3a7cff")
        capture_btn.pack(fill="x", pady=5)
        
        train_btn = ctk.CTkButton(button_frame, text="Olah Data Wajah", 
                                 command=self.train_faces,
                                 height=40, corner_radius=8,
                                 fg_color="#6c5ce7", hover_color="#5649c0")
        train_btn.pack(fill="x", pady=5)
        
        login_btn = ctk.CTkButton(button_frame, text="Login", 
                                 command=self.face_login,
                                 height=45, corner_radius=8,
                                 fg_color="#00b894", hover_color="#00a884",
                                 font=("Arial", 14, "bold"))
        login_btn.pack(fill="x", pady=(10, 5))
        
        show_hist_btn = ctk.CTkButton(button_frame, text="Histogram", 
                                    command=self.show_histogram,
                                    height=40, corner_radius=8,
                                    fg_color="#e17055", hover_color="#d63031")
        show_hist_btn.pack(fill="x", pady=5)
        
        footer_label = ctk.CTkLabel(right_frame, 
                                   text="© 2025 Riyan Wardhana",
                                   font=("Arial", 10), 
                                   text_color="gray")
        footer_label.pack(side="bottom", pady=10)

    def show_histogram(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, 640)
            self.cap.set(4, 480)
            self.update_camera_feed()
        
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            colors = ('r', 'g', 'b')
            channel_names = ('Red', 'Green', 'Blue')
            hist_data = {}
            for i, color in enumerate(colors):
                hist = cv2.calcHist([frame_rgb], [i], None, [256], [0, 256])
                hist_data[channel_names[i]] = hist.flatten()
            
            print("\n=== Histogram Values ===")
            print("Intensity\tRed\tGreen\tBlue")
            print("-" * 40)
            for intensity in range(0, 256, 12):  
                red_val = int(hist_data['Red'][intensity])
                green_val = int(hist_data['Green'][intensity])
                blue_val = int(hist_data['Blue'][intensity])
                print(f"{intensity}\t\t{red_val}\t{green_val}\t{blue_val}")
            
            print("\n=== Summary ===")
            for channel in channel_names:
                channel_data = hist_data[channel]
                print(f"{channel}: Max={int(np.max(channel_data))}, Min={int(np.min(channel_data))}, Avg={int(np.mean(channel_data))}")
            
            hist_window = ctk.CTkToplevel(self)
            hist_window.title("Color Histogram")
            hist_window.geometry("600x400")
            
            plt.figure(figsize=(6, 4))
            for i, color in enumerate(colors):
                plt.plot(hist_data[channel_names[i]], color=color)
                plt.xlim([0, 256])
            
            plt.title('Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            
            temp_file = "temp_hist.png"
            plt.savefig(temp_file)
            plt.close()
            
            hist_img = ctk.CTkImage(Image.open(temp_file), size=(580, 380))
            hist_label = ctk.CTkLabel(hist_window, image=hist_img, text="")
            hist_label.pack(padx=10, pady=10)
            
            os.remove(temp_file)
    
    def update_camera_feed(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk
        self.after_id = self.after(10, self.update_camera_feed)
    
    def take_face_samples(self):
        if not self.nama.get() or not self.nim.get():
            messagebox.showerror("Gagal", "Nama dan NIM Harus Terisi!")
            return
            
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, 640)
            self.cap.set(4, 480)
            self.update_camera_feed()
        face_id = f"{self.nim.get()}_{self.nama.get()}"
        
        if not os.path.exists("ambilWajah"):
            os.makedirs("ambilWajah")
        self.status_label.configure(text="Proses Ambil Data Wajah, Lihat Kamera yaa", 
                                  text_color="#4e8cff")
        
        count = 0
        def capture_samples():
            nonlocal count
            if count >= 30:
                self.status_label.configure(text=f"Pengambilan gambar selesai {self.nama.get()}", 
                                          text_color="#00b894")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                if self.after_id:
                    self.after_cancel(self.after_id)
                return
                
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    count += 1
                    cv2.imwrite(f"ambilWajah/{face_id}_{count}.jpg", gray[y:y+h, x:x+w])
                
            self.after(100, capture_samples)
        
        capture_samples()
    
    def train_faces(self):
        path = 'ambilWajah'
        if not os.path.exists(path):
            messagebox.showerror("Error", "Folder sampel wajah tidak ditemukan!")
            return
        face_samples = []
        ids = []
        id_names = {}
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        if not image_paths:
            messagebox.showerror("Error", "Gambar Wajah tidak ditemukan!")
            return
        user_info = {}
        for image_path in image_paths:
            if os.path.isfile(image_path):
                filename = os.path.basename(image_path)
                user_id_name = filename.split('_')[0]
                user_name = '_'.join(filename.split('_')[1:]).split('.')[0]
                if user_id_name not in user_info:
                    user_info[user_id_name] = user_name
        id_map = {user_id: idx+1 for idx, user_id in enumerate(user_info.keys())}
        for image_path in image_paths:
            if os.path.isfile(image_path):
                filename = os.path.basename(image_path)
                user_id_name = filename.split('_')[0]
                PIL_img = Image.open(image_path).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                face_id = id_map[user_id_name]
                faces = self.face_detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(face_id)
                    id_names[face_id] = user_info[user_id_name]
        if len(ids) == 0:
            messagebox.showerror("Error", "Tidak ada wajah yang terdeteksi dalam gambar!")
            return
        self.recognizer.train(face_samples, np.array(ids))
        if not os.path.exists('trainer'):
            os.makedirs('trainer')
        self.recognizer.write('trainer/trainer.yml')
        with open('trainer/id_names.pickle', 'wb') as f:
            pickle.dump(id_names, f)
        self.status_label.configure(text=f"Olah Data Selesai {len(set(ids))} Data wajah terolah.", 
                                  text_color="#00b894")
    def face_login(self):
        if not os.path.exists('trainer/trainer.yml'):
            messagebox.showerror("Error", "Olah Data Tidak Ditemukan. Olah data terlebih dahulu!")
            return
        try:
            with open('trainer/id_names.pickle', 'rb') as f:
                id_names = pickle.load(f)
        except:
            messagebox.showerror("Error", "Name mapping file not found!")
            return
            
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, 640)
            self.cap.set(4, 480)
            self.update_camera_feed()
        minW = 0.1 * self.cap.get(3)
        minH = 0.1 * self.cap.get(4)
        self.status_label.configure(text="Mengenali wajah...", text_color="#4e8cff")
        recognized = False
        recognized_name = ""
        access_denied = False
        
        def recognize_face():
            nonlocal recognized, recognized_name, access_denied
            if recognized or access_denied:
                return
            ret, img = self.cap.read()
            if not ret:
                return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                confidence_percent = round(100 - confidence)
                confidence_text = f"  {confidence_percent}%"
                
                if confidence < 50:  
                    recognized_name = id_names.get(id, "Unknown")
                    recognized = True
                    status_message = f"Login Berhasil! Selamat Datang {recognized_name}"
                    status_color = "#00b894"
                    cv2.putText(img, recognized_name, (x+5, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, confidence_text, (x+5, y+h-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                else:
                    access_denied = True
                    recognized_name = "Akses ditolak!"
                    status_message = "Akses ditolak, wajah tidak diketahui!"
                    status_color = "#d63031"
                    confidence_text = f"  {confidence_percent}% (Terlalu rendah)"
                    cv2.putText(img, recognized_name, (x+5, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(img, confidence_text, (x+5, y+h-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk
            
            if recognized:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                if self.after_id:
                    self.after_cancel(self.after_id)
                self.status_label.configure(text=status_message, text_color=status_color)
                if "Berhasil" in status_message:
                    self.show_atm_gui(recognized_name)
            elif access_denied:
                self.status_label.configure(text=status_message, text_color=status_color)
                self.after(3000, self.reset_camera)
                access_denied = False
        
        def reset_camera():
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.after_id:
                self.after_cancel(self.after_id)
            self.status_label.configure(text="Ready", text_color="gray")
        
        recognize_face()
    
    def show_atm_gui(self, username):
        atm_window = ctk.CTkToplevel(self)
        atm_window.title(f"ATM - {username}")
        atm_window.geometry("800x600")
        atm_window.resizable(False, False)
        
        window_width = atm_window.winfo_reqwidth()
        window_height = atm_window.winfo_reqheight()
        position_right = int(atm_window.winfo_screenwidth()/2 - window_width/2)
        position_down = int(atm_window.winfo_screenheight()/2 - window_height/2)
        atm_window.geometry(f"+{position_right}+{position_down}")
        
        main_frame = ctk.CTkFrame(atm_window, corner_radius=15)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(pady=20, fill="x")
        welcome_label = ctk.CTkLabel(header_frame, 
                                    text=f"Selamat Datang, {username}",
                                    font=("Arial", 24, "bold"))
        welcome_label.pack()
        balance_label = ctk.CTkLabel(header_frame, 
                                    text="Saldo: RP. 5,000.000",
                                    font=("Arial", 16),
                                    text_color="#4e8cff")
        balance_label.pack(pady=10)
        
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=20, fill="both", expand=True)
        
        row1 = ctk.CTkFrame(button_frame, fg_color="transparent")
        row1.pack(fill="x", pady=10)
        
        check_balance_btn = ctk.CTkButton(row1, text="Cek Saldo", 
                                        width=200, height=60,
                                        corner_radius=10,
                                        fg_color="#6c5ce7", hover_color="#5649c0",
                                        font=("Arial", 14))
        check_balance_btn.pack(side="left", padx=10, expand=True)
        
        withdraw_btn = ctk.CTkButton(row1, text="Tarik Tunai", 
                                    width=200, height=60,
                                    corner_radius=10,
                                    fg_color="#00b894", hover_color="#00a884",
                                    font=("Arial", 14))
        withdraw_btn.pack(side="left", padx=10, expand=True)
        
        row2 = ctk.CTkFrame(button_frame, fg_color="transparent")
        row2.pack(fill="x", pady=10)
        
        deposit_btn = ctk.CTkButton(row2, text="Masukkan Uang", 
                                  width=200, height=60,
                                  corner_radius=10,
                                  fg_color="#0984e3", hover_color="#0767b3",
                                  font=("Arial", 14))
        deposit_btn.pack(side="left", padx=10, expand=True)
        
        transfer_btn = ctk.CTkButton(row2, text="Transfer", 
                                    width=200, height=60,
                                    corner_radius=10,
                                    fg_color="#fd79a8", hover_color="#e84393",
                                    font=("Arial", 14))
        transfer_btn.pack(side="left", padx=10, expand=True)
        
        row3 = ctk.CTkFrame(button_frame, fg_color="transparent")
        row3.pack(fill="x", pady=20)
        
        logout_btn = ctk.CTkButton(row3, text="Logout", 
                                  width=200, height=50,
                                  corner_radius=10,
                                  fg_color="#d63031", hover_color="#b33939",
                                  font=("Arial", 14, "bold"),
                                  command=atm_window.destroy)
        logout_btn.pack(pady=20)
        
        footer_label = ctk.CTkLabel(main_frame, 
                                   text="For assistance, please contact your bank",
                                   font=("Arial", 10), 
                                   text_color="gray")
        footer_label.pack(side="bottom", pady=10)

if __name__ == "__main__":
    app = FaceLoginSystem()
    app.mainloop()