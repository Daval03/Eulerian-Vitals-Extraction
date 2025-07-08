"""
Klient GUI for Vitalparametermonitor

Denne modulen implementerer et grafisk brukergrensesnitt som kobles til en Flask-server for å:
- Starte/stoppe estimering av vitalparametere
- Vise verdier for hjertefrekvens og respirasjonsfrekvens
- Visualisere videostrøm for kalibrering

Kommunikasjon med serveren skjer via HTTP-forespørsler til endepunktene:
- /start: Starter estimeringsprosessen
- /stop: Stopper prosessen
- /vital-signs: Henter de sist beregnede verdiene
- /calibrate: Videostrøm for kalibrering

Hovedkomponenter:
- VitalSignsClient: Hovedklasse som håndterer grensesnittet og kommunikasjonen
- Separate tråder for:
  * Kontinuerlig overvåking av vitalparametere
  * Visualisering av videostrøm

Bruk:
    python kunde.py

Konfigurasjon:
    Serveradressen konfigureres i `client_config.ini`.
    Hvis filen ikke finnes, opprettes den med standardverdier.
"""

import tkinter as tk
import threading
import requests
import time
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import ttk, messagebox
import configparser
import os
import socket

# Konfigurasjonsfil og Discovery-innstillinger
CONFIG_FILE = "client_config.ini"
DEFAULT_BASE_URL = "http://127.0.0.1:5000" # Fallback hvis ingen server blir funnet eller konfigurert
DISCOVERY_PORT = 50001
DISCOVERY_TIMEOUT = 3 # Sekunder å lytte etter server-broadcasts

def discover_server():
    """Lytter etter server-broadcasts på UDP for å finne serverens IP og port."""
    print(f"Søker etter servere på port {DISCOVERY_PORT} i {DISCOVERY_TIMEOUT} sekunder...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('', DISCOVERY_PORT)) # Bind til alle grensesnitt
    except OSError as e:
        print(f"Kunne ikke binde til port {DISCOVERY_PORT} for serveroppdagelse: {e}. Sjekk om en annen prosess bruker porten.")
        return None

    sock.settimeout(DISCOVERY_TIMEOUT)

    servers_found = []
    try:
        while True: # Fortsett å lytte til timeout
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8')
            if message.startswith("VITAL_SIGN_SERVER_INFO:"):
                parts = message.split(':')
                if len(parts) == 3:
                    server_ip = parts[1]
                    server_port = parts[2]
                    server_url = f"http://{server_ip}:{server_port}"
                    if server_url not in servers_found:
                        servers_found.append(server_url)
                    print(f"Fant server: {server_url} fra {addr[0]}")
    except socket.timeout:
        print("Serveroppdagelse timeout.")
    except Exception as e:
        print(f"Feil under serveroppdagelse: {e}")
    finally:
        sock.close()

    if not servers_found:
        print("Ingen servere funnet via automatisk oppdagelse.")
        return None

    # For enkelhets skyld, bruk den første serveren som ble funnet.
    # En mer avansert løsning ville latt brukeren velge.
    print(f"Bruker den første oppdagede serveren: {servers_found[0]}")
    return servers_found[0]

def get_base_url():
    """Henter serverens basis-URL: Prøver først discovery, deretter konfigurasjonsfil, så standard."""
    discovered_url = discover_server()
    if discovered_url:
        return discovered_url

    print(f"Prøver å hente serveradresse fra konfigurasjonsfil: {CONFIG_FILE}")
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        try:
            config.read(CONFIG_FILE)
            base_url = config.get('server', 'address', fallback=None) # Ikke bruk fallback her ennå
            if base_url:
                if not base_url.startswith("http://") and not base_url.startswith("https://"):
                    print(f"Advarsel: Serveradressen '{base_url}' i config mangler protokoll. Bruker http.")
                    base_url = f"http://{base_url}"
                print(f"Bruker serveradresse fra konfigurasjonsfil: {base_url}")
                return base_url
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            print(f"Problem med konfigurasjonsfilen '{CONFIG_FILE}': {e}.")
        except Exception as e:
            print(f"Uventet feil ved lesing av konfigurasjonsfil '{CONFIG_FILE}': {e}.")

    print(f"Ingen server funnet via discovery eller i konfigurasjonsfil. Bruker standard URL: {DEFAULT_BASE_URL}")
    # Opprett en standard konfigurasjonsfil hvis den ikke finnes og ingen server ble oppdaget
    if not os.path.exists(CONFIG_FILE) and not discovered_url:
        try:
            config['server'] = {'address': DEFAULT_BASE_URL,
                                'comment': 'Du kan endre adressen ovenfor til IP-adressen eller vertsnavnet til serveren din, eller la den stå tom for automatisk oppdagelse.'}
            with open(CONFIG_FILE, 'w') as configfile:
                config.write(configfile)
            print(f"Opprettet en standard konfigurasjonsfil: {CONFIG_FILE}")
        except Exception as e:
            print(f"Kunne ikke opprette standard konfigurasjonsfil: {e}")
    return DEFAULT_BASE_URL

BASE_URL = get_base_url()

class VitalSignsClient:
    """
    Hovedklasse for klient-GUI-en for overvåking av vitalparametere.
    
    Attributter:
        root: Hovedvindu i Tkinter
        is_running: Flagg for estimeringsstatus
        stop_event: Hendelse for å kontrollere tråder
        video_running: Flagg for videostatus
        video_label: Widget for å vise videostrømmen
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Vitalparametermonitor")
        self.root.geometry("650x750")
        self.root.configure(bg="#f0f4f8")
        self.root.resizable(False, False)
        # Kontrollflagg
        self.is_running = False
        self.stop_event = threading.Event()
        self.video_running = False
        
        # Moderne stil
        self.style = ttk.Style()
        self.configure_style()
        
        # Elementer for videoen
        self.video_label = tk.Label(self.root)
        self.video_thread = None

        # Opprett brukergrensesnitt
        self.create_ui()

    def configure_style(self):
        # Konfigurer stiler for et mer moderne utseende
        self.style.theme_use('clam')
        # Stil for knapper
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
        # Stil for hovedramme
        self.style.configure('Main.TFrame', 
            background='#f0f4f8')

    def create_ui(self):
        # Hovedramme
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Serveradresse-etikett
        self.server_address_label = tk.Label(
            main_frame,
            text=f"Server: {BASE_URL}",
            font=("Arial", 8),
            bg="#f0f4f8",
            fg="#555"
        )
        self.server_address_label.pack(pady=(0,5), anchor='w')

        # Tittel
        title_label = tk.Label(
            main_frame, 
            text="Vitalparametermonitor",
            font=("Arial", 16, "bold"),
            bg="#f0f4f8",
            fg="#2c3e50"
        )
        title_label.pack(pady=(0,20))
        
        # Beholder for vitalparametere
        vitals_frame = tk.Frame(
            main_frame, 
            bg="white", 
            borderwidth=1, 
            relief=tk.RAISED)
        vitals_frame.pack(fill=tk.X, pady=10)
        
        # Hjertefrekvens
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
            text="Hjertefrekvens: -- slag/min",
            font=("Arial", 12),
            bg="white",
            anchor="w")
        self.heart_rate_label.pack(side=tk.LEFT, expand=True)
        
        # Respirasjonsfrekvens
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
            text="Respirasjonsfrekvens: -- pust/min",
            font=("Arial", 12),
            bg="white",
            anchor="w")
        self.resp_rate_label.pack(side=tk.LEFT, expand=True)

        # Knapper
        button_frame = tk.Frame(main_frame, bg="#f0f4f8")
        button_frame.pack(fill=tk.X, pady=20)

        # Startknapp
        self.start_button = ttk.Button(
            button_frame, 
            text="Start Estimering",
            command=self.start_estimation,
            style='Start.TButton')
        self.start_button.pack(side=tk.LEFT, expand=True, padx=10)

        # Stoppknapp
        self.stop_button = ttk.Button(
            button_frame, 
            text="Stopp Estimering",
            command=self.stop_estimation,
            style='Stop.TButton')
        self.stop_button.pack(side=tk.RIGHT, expand=True, padx=10)
        
        # Knapp for video
        self.video_button = ttk.Button(
            button_frame,
            text="Vis/Skjul Video",
            command=self.toggle_video_stream,
            style='Start.TButton')
        self.video_button.pack(side=tk.LEFT, expand=True, padx=10)

        # Statusetikett
        self.status_label = tk.Label(
            main_frame, 
            text="Status: Stoppet",
            font=("Arial", 10),
            bg="#f0f4f8",
            fg="#7f8c8d")
        self.status_label.pack(pady=10)
        
        # Etikett for videoen
        self.video_label.pack(pady=10)

    def toggle_video_stream(self):
        if self.video_running:
            self.stop_video_stream()
            self.video_button.config(text="Vis Video")
        else:
            self.start_video_stream()
            self.video_button.config(text="Skjul Video")

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
                # Kall start-endepunktet
                response = requests.get(f"{BASE_URL}/start")
                if response.status_code == 200:
                    self.is_running = True
                    self.status_label.config(
                        text="Status: Estimerer",
                        fg="#27ae60")
                    # Start tråd for å overvåke vitalparametere
                    self.monitoring_thread = threading.Thread(
                        target=self.monitor_vital_signs, 
                        daemon=True)
                    self.stop_event.clear()
                    self.monitoring_thread.start()
                else:
                    messagebox.showerror("Feil", f"Kunne ikke starte estimering: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                messagebox.showerror("Tilkoblingsfeil", f"Kunne ikke koble til serveren på {BASE_URL}. Kontroller serveradressen og at serveren kjører.")
            except requests.exceptions.Timeout:
                messagebox.showerror("Tidsavbrudd", f"Forespørselen til {BASE_URL}/start timet ut. Serveren svarer kanskje ikke.")
            except requests.exceptions.RequestException as e:
                messagebox.showerror("Nettverksfeil", f"En uventet nettverksfeil oppstod: {e}")

    def stop_estimation(self):
        if self.is_running:
            try:
                # Kall stopp-endepunktet
                response = requests.get(f"{BASE_URL}/stop", timeout=5) # Legg til timeout
                if response.status_code == 200:
                    self.is_running = False
                    self.stop_event.set()
                    self.status_label.config(
                        text="Status: Stoppet",
                        fg="#7f8c8d")
                    # Gjenopprett etiketter
                    self.heart_rate_label.config(
                        text="Hjertefrekvens: -- slag/min")
                    self.resp_rate_label.config(
                        text="Respirasjonsfrekvens: -- pust/min")
                else:
                    messagebox.showerror("Feil", f"Kunne ikke stoppe estimering: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                messagebox.showerror("Tilkoblingsfeil", f"Kunne ikke koble til serveren på {BASE_URL} for å stoppe. Kontroller serverforbindelsen.")
            except requests.exceptions.Timeout:
                messagebox.showerror("Tidsavbrudd", f"Forespørselen til {BASE_URL}/stop timet ut.")
            except requests.exceptions.RequestException as e:
                messagebox.showerror("Nettverksfeil", f"En uventet nettverksfeil oppstod ved stopping: {e}")

    def monitor_vital_signs(self):
        retry_delay = 1  # Start med 1 sekund forsinkelse
        max_retries = 5
        retries = 0

        while not self.stop_event.is_set():
            try:
                # Hent vitalparametere
                response = requests.get(f"{BASE_URL}/vital-signs", timeout=2) # Kortere timeout for hyppige kall
                if response.status_code == 200:
                    data = response.json()
                    # Oppdater etikettene i hovedtråden
                    self.root.after(0, self.update_vital_signs, data)
                    retries = 0 # Nullstill antall forsøk ved suksess
                    retry_delay = 1 # Nullstill forsinkelse ved suksess
                elif response.status_code == 404: # Serveren har ikke data ennå
                    # Ikke en feil, bare vent og prøv igjen
                    pass
                else:
                    print(f"Feil ved henting av vitalparametere: {response.status_code}")
                    # Vurder å vise en ikke-blokkerende status til brukeren her

                time.sleep(1) # Normal forsinkelse mellom forespørsler

            except requests.exceptions.ConnectionError:
                retries += 1
                if retries >= max_retries:
                    self.root.after(0, self.handle_connection_error, "Mistet tilkobling etter flere forsøk.")
                    break
                print(f"Tilkoblingsfeil under overvåking. Prøver igjen om {retry_delay} sekunder... (Forsøk {retries}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30) # Doble forsinkelse, maks 30 sek
            except requests.exceptions.Timeout:
                retries += 1
                if retries >= max_retries:
                    self.root.after(0, self.handle_connection_error, "Serveren sluttet å svare (timeout).")
                    break
                print(f"Timeout under overvåking. Prøver igjen om {retry_delay} sekunder... (Forsøk {retries}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
            except requests.exceptions.RequestException as e:
                self.root.after(0, self.handle_connection_error, f"Uventet nettverksfeil: {e}")
                break
            except Exception as e: # Fang andre uventede feil
                print(f"Uventet feil i monitor_vital_signs: {e}")
                self.root.after(0, self.handle_connection_error, f"En intern feil oppstod: {e}")
                break


    def update_vital_signs(self, data):
        # Oppdater etiketter med mottatte data
        heart_rate = data.get('heart_rate', '-')
        resp_rate = data.get('respiratory_rate', '-')
        self.heart_rate_label.config(text=f"Hjertefrekvens: {heart_rate:.2f} slag/min")
        self.resp_rate_label.config(text=f"Respirasjonsfrekvens: {resp_rate:.2f} pust/min")

    def handle_connection_error(self, message="Tilkobling til serveren mistet"):
        # Metode for å håndtere tilkoblingsfeil
        if self.is_running or not self.stop_event.is_set(): # Bare vis feil hvis vi ikke allerede har stoppet
            self.is_running = False
            self.stop_event.set() # Sørg for at overvåkingstråden stopper
            self.status_label.config(
                text=f"Status: Tilkoblingsfeil",
                fg="#e74c3c"
            )
            # Gjenopprett etiketter til "--" hvis de ikke allerede er det
            if "--" not in self.heart_rate_label.cget("text"):
                 self.heart_rate_label.config(text="Hjertefrekvens: -- slag/min")
            if "--" not in self.resp_rate_label.cget("text"):
                self.resp_rate_label.config(text="Respirasjonsfrekvens: -- pust/min")

            messagebox.showerror("Tilkoblingsfeil", message)


def main():
    # Informer brukeren om hvilken server som brukes
    print(f"Kobler til server på: {BASE_URL}")
    # TODO: Vurder å legge til en liten etikett i GUI-en som viser BASE_URL.
    root = tk.Tk()
    app = VitalSignsClient(root)
    root.mainloop()

if __name__ == "__main__":
    main()