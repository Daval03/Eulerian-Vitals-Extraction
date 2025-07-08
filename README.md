# Eulerian Vitals Extraction - Vitalparametermonitor

## 📌 Beskrivelse (Description)
Dette prosjektet implementerer et **ikke-invasivt system** for å estimere **hjertefrekvens (HR)** og **respirasjonsfrekvens (RR)** ved hjelp av en datamaskin med webkamera (f.eks. en Raspberry Pi) og dets kamera. Systemet anvender **Eulerian Video Magnification (EVM)** for å forsterke subtile fargeendringer i ansiktet og bevegelser, trekker ut fysiologiske signaler, og beregner vitale tegn gjennom signalbehandling.

Systemet består av:
-   ✅ **Klient-Server Arkitektur**: En Flask backend (tjener) for videoprosessering og en Tkinter GUI (klient) for fjernvisualisering og kontroll. Klienten og tjeneren er skrevet i Python.
-   ✅ **Ansiktsgjenkjenning**: Python-serveren bruker OpenCV for å detektere ansikter og definere en Region of Interest (ROI).
-   ✅ **Signalbehandling**: Implementerer EVM, PCA, og båndpassfiltrering for å trekke ut hjerte- og respirasjonsfrekvenser.
-   ✅ **Grafisk Brukergrensesnitt (GUI)**: Viser sanntids vitalparametere med start/stopp-kontroller og video-feed for kalibrering. Brukergrensesnittet er oversatt til Norsk.

## 🎯 Formål (Objective)
Utvikle et lavkostnads, portabelt og kontaktløst system for fjernovervåking av vitalparametere, egnet for applikasjoner innen telemedisin, digital helse og pasientovervåking.

---

## 🛠️ Teknologier & Maskinvare (Technologies & Hardware)
### 🔹 Maskinvare (Hardware)
-   En datamaskin med Python 3.7+ (f.eks. PC, Mac, Raspberry Pi 4 - 4GB RAM anbefalt for Pi).
-   Webkamera (innebygd eller USB), som gjenkjennes av OpenCV (vanligvis som indeks 0).

### 🔹 Nøkkelbiblioteker (Key Libraries - Python)
-   **OpenCV-Python**: Videoprosessering og ansiktsgjenkjenning.
-   **NumPy & SciPy**: Signalbehandling og numeriske operasjoner.
-   **Flask & Flask-CORS**: Backend API-server.
-   **Tkinter**: Klient GUI.
-   **Pillow (PIL)**: Bildebehandling for GUI.
-   **Requests**: HTTP-forespørsler fra klient til server.
-   **ConfigParser**: For håndtering av klientkonfigurasjon.

---

## 🚀 Installasjon & Bruk (Installation & Usage)

### Forutsetninger (Prerequisites)
Installer nødvendige Python-biblioteker. Det anbefales på det sterkeste å bruke et virtuelt miljø for å unngå konflikter med andre Python-prosjekter.

1.  **Opprett et virtuelt miljø (anbefalt):**
    ```bash
    python -m venv venv
    ```
2.  **Aktiver det virtuelle miljøet:**
    -   På Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    -   På macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
3.  **Installer nødvendige pakker:**
    Naviger til rotmappen til prosjektet (hvor denne README.md-filen er) og kjør:
    ```bash
    pip install opencv-python numpy scipy flask flask-cors pillow requests configparser
    ```

### Kjøre applikasjonen (Python Klient/Server)

Systemet består av en server (`Python/Code/servidor.py`) som utfører vitalparameterestimeringen og en klient (`Python/Code/cliente.py`) som viser resultatene. Begge må kjøres fra `Python/Code`-mappen.

**1. Start Serveren:**
Naviger til `Python/Code`-mappen i terminalen (med det virtuelle miljøet aktivert) og kjør:
```bash
python servidor.py
```
Serveren vil starte og begynne å kringkaste sin tilstedeværelse på det lokale nettverket for automatisk oppdagelse av klienten. Den vil skrive ut sin lokale IP-adresse og porten den lytter på (standard er `http://<din-ip>:5000`). Noter deg denne adressen hvis automatisk oppdagelse feiler.

**2. Start Klienten:**
Åpne en *ny* terminal (eller en ny fane i din eksisterende terminal), naviger til `Python/Code`-mappen, og aktiver det virtuelle miljøet på samme måte som for serveren hvis du ikke allerede har gjort det i den nye terminalen/fanen. Kjør deretter:
```bash
python cliente.py
```

**Klientens Serverforbindelse:**
Klienten kobler til serveren på følgende måte (prioritert rekkefølge):
1.  **Automatisk Oppdagelse**: Ved oppstart vil klienten først lytte i 3 sekunder for servere som kringkaster sin adresse på det lokale nettverket (via UDP på port 50001). Hvis en eller flere servere blir funnet, brukes den første som blir oppdaget.
2.  **Konfigurasjonsfil (`client_config.ini`)**: Hvis ingen server oppdages automatisk, vil klienten se etter en `client_config.ini`-fil i `Python/Code`-mappen. Hvis denne filen eksisterer og inneholder en gyldig serveradresse under `[server]` -> `address`, vil den bli brukt.
    Eksempel på `client_config.ini`:
    ```ini
    [server]
    address = http://192.168.1.10:5000
    ; Du kan endre adressen ovenfor til IP-adressen eller vertsnavnet til serveren din.
    ```
3.  **Standardadresse**: Hvis verken automatisk oppdagelse lykkes eller en gyldig konfigurasjonsfil finnes (eller adressen i filen er ugyldig), vil klienten falle tilbake til standardadressen `http://127.0.0.1:5000` (localhost).
4.  **Opprettelse av `client_config.ini`**: Hvis `client_config.ini` ikke finnes når klienten starter, og automatisk oppdagelse ikke finner en server, vil en standard `client_config.ini` bli opprettet med `http://127.0.0.1:5000` som adresse. Du kan deretter redigere denne filen manuelt.

**Bruk av GUI (Klientprogrammet):**
Når klienten starter og kobler seg til serveren, vil du se følgende:
-   **Serveradresse**: Øverst til venstre vises adressen til serveren klienten er koblet til.
-   **"Start Estimering"**: Klikk denne knappen for å starte prosessen for estimering av vitalparametere på serveren.
-   **"Stopp Estimering"**: Klikk denne knappen for å stoppe estimeringsprosessen.
-   **"Vis/Skjul Video"**: Viser eller skjuler sanntids videostrøm fra serverens kamera. Dette er nyttig for å sjekke at ansiktet ditt er godt synlig for kameraet.
-   **Hjertefrekvens**: Viser estimert hjertefrekvens i slag per minutt (slag/min).
-   **Respirasjonsfrekvens**: Viser estimert respirasjonsfrekvens i pust per minutt (pust/min).
-   **Status**: Viser nåværende status for applikasjonen (f.eks. "Stoppet", "Estimerer", "Tilkoblingsfeil").

### C++ Komponenter
Prosjektet inneholder også C++ kode i `Cpp/Code`-mappen (`Vital_Estimator.cpp`, `Face_Detector.cpp`, etc.). Denne delen er en separat implementering for vitalparameterestimering og er **ikke direkte integrert** med Python klient/server-arkitekturen som beskrevet ovenfor. Den bruker en annen metode for ansiktsgjenkjenning (Caffe-modell) og har sin egen konfigurasjon (`Cpp/Code/Config.h`).
-   For å kompilere og kjøre C++ delen (forutsetter at du har g++, OpenCV C++ og Eigen3 bibliotekene installert):
    ```bash
    cd Cpp/Code
    make
    ./Vital_Estimator
    ```
    Denne C++ applikasjonen vil forsøke å lese fra et kamera eller en videofil som spesifisert i `Config.h`. De nylige oversettelsene og robusthetsforbedringene har **primært fokusert på Python-applikasjonen**.

---

## 📝 Viktige Filer (Key Files)
-   `Python/Code/servidor.py`: Flask server-applikasjonen (backend).
-   `Python/Code/cliente.py`: Tkinter GUI klient-applikasjonen (frontend).
-   `Python/Code/client_config.ini`: (Opprettes/leses av klienten) Konfigurasjonsfil for klienten.
-   `Python/Code/vital_signs_estimator.py`: Kjernelogikk for estimering av vitalparametere (brukes av serveren).
-   `Python/Code/face_detector.py`: Ansiktsgjenkjenningsmodul for Python-serveren.
-   `Python/Code/signal_processing.py`: Signalbehandlingsfunksjoner for Python-serveren.
-   `Python/Code/config.py`: Generelle konfigurasjonsparametere for Python-estimatoren.
-   `Cpp/Code/Vital_Estimator.cpp`: Hovedfil for den separate C++ estimeringslogikken.
-   `Cpp/Code/Config.h`: Konfigurasjonsfil for C++ delen.

---

Lykke til! (Good luck!)
