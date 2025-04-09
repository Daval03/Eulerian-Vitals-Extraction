import tkinter as tk
import threading
import requests
import time
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import ttk, messagebox

# URL base del servidor
BASE_URL = "http://192.168.101.9:5000"

class VitalSignsClient:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitor de Signos Vitales")
        self.root.geometry("650x750")
        self.root.configure(bg="#f0f4f8")
        self.root.resizable(False, False)
        # Banderas de control
        self.is_running = False
        self.stop_event = threading.Event()
        self.video_running = False
        
        # Estilo moderno
        self.style = ttk.Style()
        self.configure_style()
        
        # Elementos para el video
        self.video_label = tk.Label(self.root)
        self.video_thread = None

        # Crear interfaz
        self.create_ui()

    def configure_style(self):
        # Configurar estilos para un aspecto más moderno
        self.style.theme_use('clam')
        # Estilo para botones
        self.style.configure('Start.TButton', 
            background='#4CAF50', 
            foreground='white', 
            font=('Arial', 12, 'bold'))
        self.style.map('Start.TButton', 
            background=[('active', '#45a049')])
        self.style.configure('Stop.TButton', 
            background='#f44336', 
            foreground='white', 
            font=('Arial', 12, 'bold'))
        self.style.map('Stop.TButton', 
            background=[('active', '#d32f2f')])
        # Estilo para frame principal
        self.style.configure('Main.TFrame', 
            background='#f0f4f8')

    def create_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Título
        title_label = tk.Label(
            main_frame, 
            text="Monitor de Signos Vitales", 
            font=("Arial", 16, "bold"),
            bg="#f0f4f8",
            fg="#2c3e50"
        )
        title_label.pack(pady=(0,20))
        
        # Contenedor para signos vitales
        vitals_frame = tk.Frame(
            main_frame, 
            bg="white", 
            borderwidth=1, 
            relief=tk.RAISED)
        vitals_frame.pack(fill=tk.X, pady=10)
        
        # Frecuencia cardíaca
        heart_frame = tk.Frame(vitals_frame, bg="white")
        heart_frame.pack(fill=tk.X, padx=10, pady=5, anchor="w")
        heart_icon = tk.Label(
            heart_frame, 
            text="❤️", 
            font=("Arial", 20), 
            bg="white")
        heart_icon.pack(side=tk.LEFT, padx=(0,10))

        self.heart_rate_label = tk.Label(
            heart_frame, 
            text="Frecuencia cardíaca: -- lpm", 
            font=("Arial", 12),
            bg="white",
            anchor="w")
        self.heart_rate_label.pack(side=tk.LEFT, expand=True)
        
        # Frecuencia respiratoria
        resp_frame = tk.Frame(vitals_frame, bg="white")
        resp_frame.pack(fill=tk.X, padx=10, pady=5, anchor="w")
        resp_icon = tk.Label(
            resp_frame, 
            text="🫁", 
            font=("Arial", 20), 
            bg="white")
        resp_icon.pack(side=tk.LEFT, padx=(0,10))
        self.resp_rate_label = tk.Label(
            resp_frame, 
            text="Frecuencia respiratoria: -- rpm", 
            font=("Arial", 12),
            bg="white",
            anchor="w")
        self.resp_rate_label.pack(side=tk.LEFT, expand=True)

        # Botones
        button_frame = tk.Frame(main_frame, bg="#f0f4f8")
        button_frame.pack(fill=tk.X, pady=20)

        # Botón Iniciar
        self.start_button = ttk.Button(
            button_frame, 
            text="Iniciar Estimación", 
            command=self.start_estimation,
            style='Start.TButton')
        self.start_button.pack(side=tk.LEFT, expand=True, padx=10)

        # Botón Detener
        self.stop_button = ttk.Button(
            button_frame, 
            text="Detener Estimación", 
            command=self.stop_estimation,
            style='Stop.TButton')
        self.stop_button.pack(side=tk.RIGHT, expand=True, padx=10)
        
        # Botón para video
        self.video_button = ttk.Button(
            button_frame,
            text="Mostrar/Ocultar Video",
            command=self.toggle_video_stream,
            style='Start.TButton')
        self.video_button.pack(side=tk.LEFT, expand=True, padx=10)

        # Etiqueta de estado
        self.status_label = tk.Label(
            main_frame, 
            text="Estado: Detenido", 
            font=("Arial", 10),
            bg="#f0f4f8",
            fg="#7f8c8d")
        self.status_label.pack(pady=10)
        
        # Etiqueta para el video
        self.video_label.pack(pady=10)

    def toggle_video_stream(self):
        if self.video_running:
            self.stop_video_stream()
            self.video_button.config(text="Mostrar Video")
        else:
            self.start_video_stream()
            self.video_button.config(text="Ocultar Video")

    def start_video_stream(self):
        if not self.video_running:
            self.video_running = True
            self.video_thread = threading.Thread(target=self.video_stream, daemon=True)
            self.video_thread.start()

    def stop_video_stream(self):
        self.video_running = False

    def video_stream(self):
        cap = cv2.VideoCapture(f"{BASE_URL}/calibrate")
        while self.video_running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        cap.release()
        self.video_label.config(image='')

    def start_estimation(self):
        if not self.is_running:
            try:
                # Llamar al endpoint de inicio
                response = requests.get(f"{BASE_URL}/start")
                if response.status_code == 200:
                    self.is_running = True
                    self.status_label.config(
                        text="Estado: Estimando", 
                        fg="#27ae60")
                    # Iniciar hilo para monitorear signos vitales
                    self.monitoring_thread = threading.Thread(
                        target=self.monitor_vital_signs, 
                        daemon=True)
                    self.stop_event.clear()
                    self.monitoring_thread.start()
                else:
                    messagebox.showerror("Error", f"No se pudo iniciar: {response.json().get('status')}")
            except requests.exceptions.RequestException as e:
                messagebox.showerror("Error de Conexión", str(e))

    def stop_estimation(self):
        if self.is_running:
            try:
                # Llamar al endpoint de detención
                response = requests.get(f"{BASE_URL}/stop")
                if response.status_code == 200:
                    self.is_running = False
                    self.stop_event.set()
                    self.status_label.config(
                        text="Estado: Detenido", 
                        fg="#7f8c8d")
                    # Restaurar etiquetas
                    self.heart_rate_label.config(
                        text="Frecuencia cardíaca: -- lpm")
                    self.resp_rate_label.config(
                        text="Frecuencia respiratoria: -- rpm")
                else:
                    messagebox.showerror("Error", f"No se pudo detener: {response.json().get('status')}")
            except requests.exceptions.RequestException as e:
                messagebox.showerror("Error de Conexión", str(e))

    def monitor_vital_signs(self):
        while not self.stop_event.is_set():
            try:
                # Obtener los signos vitales
                response = requests.get(f"{BASE_URL}/vital-signs")
                if response.status_code == 200:
                    data = response.json()
                    # Actualizar las etiquetas en el hilo principal
                    self.root.after(0, self.update_vital_signs, data)
                # Esperar un poco antes de la próxima consulta
                time.sleep(1)
            except requests.exceptions.RequestException:
                # Manejar errores de conexión
                self.root.after(0, self.handle_connection_error)
                break

    def update_vital_signs(self, data):
        # Actualizar etiquetas con datos recibidos
        heart_rate = data.get('heart_rate', '-')
        resp_rate = data.get('respiratory_rate', '-')
        self.heart_rate_label.config(text=f"Frecuencia cardíaca: {heart_rate:.2f} lpm")
        self.resp_rate_label.config(text=f"Frecuencia respiratoria: {resp_rate:.2f} rpm")

    def handle_connection_error(self):
        # Método para manejar errores de conexión
        self.is_running = False
        self.stop_event.set()
        self.status_label.config(
            text="Estado: Error de Conexión", 
            fg="#e74c3c"
        )
        messagebox.showerror("Error", "Pérdida de conexión con el servidor")

def main():
    root = tk.Tk()
    app = VitalSignsClient(root)
    root.mainloop()

if __name__ == "__main__":
    main()