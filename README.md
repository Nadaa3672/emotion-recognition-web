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

| | spiegazione<br>con propagazione indietro | spiegazione<br>nella stessa passata | senza<br>spiegazione |
|---|---:|---:|---:|
| Costo della spiegazione | 44,7 ms | **0,4 ms** | — |
| Latenza mediana | 78,4 ms | **34,0 ms** | 24,1 ms |
| Latenza al 95° percentile | 132,3 ms | 108,8 ms | 100,0 ms |
| Richieste al secondo | 12,8 | **29,4** | 41,4 |

Prima richiesta a freddo: ~2,4 s. Dimensione del modello: 380 KB.

### L'ottimizzazione della spiegazione

Nella formulazione generale, Grad-CAM richiede una propagazione all'indietro
attraverso la rete e diventa lo stadio più costoso della catena: 44,7 ms contro i
24 ms della predizione.

Questa architettura termina però con un global average pooling seguito da un unico
strato lineare. Il gradiente del logit rispetto alle attivazioni dell'ultimo blocco
convoluzionale è allora costante e pari al peso dello strato lineare diviso per il
numero di posizioni spaziali; il peso Grad-CAM di ciascun canale coincide quindi con
il peso del classificatore, a meno di un fattore che la normalizzazione della mappa
elimina. La spiegazione si ottiene dalla **medesima passata in avanti** che produce
la predizione.

L'equivalenza è verificata numericamente: la differenza massima fra le due mappe è
dell'ordine di 10⁻⁷, cioè l'errore di arrotondamento in virgola mobile.

Il costo della spiegazione si riduce di **104 volte**, la latenza complessiva del
**57%**, e la capacità di servizio passa da 12,8 a 29,4 richieste al secondo. La
trasparenza smette di essere un compromesso: il sistema spiega ogni decisione senza
pagare nulla di apprezzabile.

## Installazione ed esecuzione

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

L'applicazione si apre su `http://localhost:8501`.

Per riprodurre le misure e la valutazione:

```bash
python -m benchmark.latency          # prestazioni del servizio
python -m evaluation.evaluate        # metriche del classificatore
python -m evaluation.figure          # figure dei risultati
```

## Struttura del repository

```
streamlit_app.py        interfaccia web
app.py                  interfaccia alternativa (Gradio), per uso locale
src/config.py           parametri audio, etichette, mappatura sulla valenza
src/preprocessing.py    catena di pre-elaborazione del segnale
src/model.py            architettura della rete e caricamento del checkpoint
src/gradcam.py          spiegazione per localizzazione (due implementazioni)
src/inference.py        livello di servizio, con cronometraggio degli stadi
src/viz.py              figure
benchmark/latency.py    misura delle prestazioni del servizio
evaluation/             valutazione sperimentale del classificatore
models/best_cnn.pt      pesi del modello addestrato
results/                risultati numerici e figure
```

Le due interfacce sono clienti intercambiabili dello stesso livello di servizio:
tutta la logica vive in `src/inference.py`, e nessuna delle due ne contiene una riga.

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

Nada Hekal — matricola [DA COMPLETARE]
Corso: Sistemi Intelligenti per Internet
