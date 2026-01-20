Prompt Context: Agisci come un Senior System Architect esperto in MedTech e Simulazione Multifisica. Il progetto riguarda un dispositivo medico a onde d'urto focalizzate a generazione elettroidraulica.

Obiettivo: Creare un Digital Twin funzionale che integri Elettronica, Fisica dei Plasmi e Chimica dei Materiali.

1. Requisiti di Sistema (Input): * Generatore: Carica condensatore 10-30 kV. * Mezzo: Soluzione acqua + palladio (catalizzatore). * Fenomeno: Ionizzazione a cascata, formazione plasma (>10.000 K), vaporizzazione istantanea e onda d'urto.
+3

2. Moduli da Sviluppare:

A. Power Electronics Module: Simulare la scarica del condensatore attraverso un tiristore. Calcolare la corrente che attraversa il canale di plasma.
+2

B. Physics Engine (Acoustics): Modellare la propagazione dell'onda d'urto e la sua focalizzazione tramite riflettore ellissoidale su un target a distanza X.

C. Degradation & Chemistry Module: Modellare l'erosione dell'elettrodo (aumento distanza punte) e l'accumulo di ossidi/detriti metallici che alterano la conducibilità dell'acqua. Simulare l'efficienza del palladio nell'assorbire le bolle di gas.
+3

D. Control Logic (Firmware): Sviluppare un algoritmo PID o Fuzzy Logic per il meccanismo di avvicinamento automatico delle punte basato sul monitoraggio del feedback elettrico.

3. Vincoli Tecnici: * Il simulatore deve prevedere l'impatto dell'aumento della concentrazione di soluti sulla tensione superficiale e sul potenziale di rottura dielettrica.

Output richiesti: Grafico della pressione focale (MPa), efficienza della scarica in funzione del numero di impulsi, log degli interventi del motore di avvicinamento.

Istruzione Operativa: Inizia definendo le equazioni differenziali di base per la fase di formazione del plasma e la propagazione acustica non lineare.