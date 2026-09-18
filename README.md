# Riconoscimento delle emozioni dal parlato — servizio web

Progetto per il corso di **Sistemi Intelligenti per Internet** — A.A. 2025/2026
Università degli Studi Roma Tre, M.Sc. in Computer Science Engineering

---

## Cosa fa

L'utente registra la propria voce dal browser oppure carica un file audio. Il
sistema restituisce:

- la **distribuzione di probabilità** sulle otto emozioni (neutrale, calmo, felice,
  triste, arrabbiato, impaurito, disgustato, sorpreso);
- una **sintesi sull'asse della valenza** (positiva / neutra / negativa), ottenuta
  proiettando le otto classi secondo il modello circomplesso di Russell;
- la **rappresentazione tempo-frequenza** da cui la decisione è stata tratta;
- la **spiegazione Grad-CAM**, che evidenzia le regioni dello spettrogramma che
  hanno determinato la predizione;
- i **tempi di risposta** dei singoli stadi della catena.

La classificazione si basa sulla prosodia — melodia, intensità, ritmo, dinamica
spettrale — e non sul contenuto lessicale: il sistema analizza *come* una frase
viene detta, non cosa viene detto.

## Architettura

```
registrazione
    │
    ├─ pre-elaborazione   mono · 16 kHz · rimozione del silenzio · durata fissa 3 s
    ├─ rappresentazione   spettrogramma log-Mel 128 × 188  +  derivate Δ e ΔΔ
    ├─ inferenza          CNN a tre blocchi convoluzionali (94.728 parametri)
    └─ spiegazione        Grad-CAM sull'ultimo blocco convoluzionale
```

Il modello viene caricato una sola volta all'avvio e mantenuto in memoria; una
chiamata di riscaldamento paga subito i costi di inizializzazione, così che il
primo utente non attenda la compilazione a caldo delle librerie di elaborazione
del segnale.

## Prestazioni del servizio

Misure su CPU, elaborazione sequenziale (`python -m benchmark.latency`):

| | con spiegazione | senza spiegazione |
|---|---:|---:|
| Latenza mediana | 81,6 ms | 25,0 ms |
| Latenza al 95° percentile | 140,4 ms | 91,3 ms |
| Richieste al secondo | 12,3 | 40,0 |

Prima richiesta a freddo: ~2,4 s. Dimensione del modello: 380 KB.

Il calcolo della spiegazione richiede una propagazione all'indietro attraverso la
rete e costa circa **tre volte** la predizione stessa: per questo nell'interfaccia
è disattivabile.

## Installazione ed esecuzione

```bash
pip install -r requirements.txt
python app.py
```

L'applicazione si apre su `http://127.0.0.1:7860`.

## Struttura del repository

```
app.py                  applicazione web (Gradio)
src/config.py           parametri audio, etichette, mappatura sulla valenza
src/preprocessing.py    catena di pre-elaborazione del segnale
src/model.py            architettura della rete e caricamento del checkpoint
src/gradcam.py          spiegazione per localizzazione
src/inference.py        livello di servizio, con cronometraggio degli stadi
src/viz.py              figure dell'interfaccia
benchmark/latency.py    misura delle prestazioni del servizio
evaluation/             valutazione sperimentale del classificatore
models/best_cnn.pt      pesi del modello addestrato
results/                risultati numerici
```

## Dati

[RAVDESS](https://zenodo.org/record/1188976) — Ryerson Audio-Visual Database of
Emotional Speech and Song, sottoinsieme del parlato: 1.440 registrazioni di 24
attori professionisti su otto stati emotivi.

La valutazione adotta una separazione rigorosa dei parlanti: nessun attore compare
in più di una partizione (16 addestramento / 4 validazione / 4 test). Senza questa
precauzione il classificatore può ottenere accuratezze apparentemente alte
riconoscendo l'identità della voce anziché l'emozione che essa porta.

## Strumenti e librerie

Python · PyTorch · librosa · scikit-learn · SciPy · Gradio · matplotlib · NumPy

## Autrice

Nada Hekal
