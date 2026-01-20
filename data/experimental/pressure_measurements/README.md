# Misure di Pressione Sperimentali

Questa cartella è destinata a contenere i dati sperimentali di misure di pressione
effettuate con idrofono durante test ESWT.

## Formato Dati Atteso

I file dovrebbero essere in formato CSV con le seguenti colonne:

```csv
tempo_us,pressione_MPa,tensione_kV,energia_J,distanza_cm,note
0.0,0.0,20,600,17.5,inizio acquisizione
0.1,0.5,20,600,17.5,
0.2,2.3,20,600,17.5,
...
```

### Colonne Obbligatorie:
- `tempo_us`: Tempo in microsecondi dall'inizio della scarica
- `pressione_MPa`: Pressione misurata in MegaPascal

### Colonne Opzionali:
- `tensione_kV`: Tensione di carica del condensatore
- `energia_J`: Energia depositata
- `distanza_cm`: Distanza dal punto focale
- `note`: Note addizionali

## Naming Convention

Usare il seguente formato per i nomi file:
```
YYYYMMDD_tensione_energia_distanza.csv
```

Esempio: `20240115_20kV_600J_17cm.csv`

## Metadati

Creare un file `metadata.json` nella stessa cartella con informazioni su:
- Setup sperimentale
- Tipo di idrofono
- Temperatura acqua
- Conducibilità acqua
- Note sulla calibrazione
