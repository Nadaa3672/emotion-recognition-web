"""
Applicazione web per il riconoscimento delle emozioni dal parlato.

L'utente registra la voce dal browser o carica un file audio; il sistema
restituisce la distribuzione di probabilità sulle otto emozioni, una sintesi
sull'asse della valenza, la rappresentazione tempo-frequenza da cui la decisione
è stata tratta, la mappa delle regioni che l'hanno determinata e i tempi di
risposta dei singoli stadi.

Avvio locale:   python app.py
"""
import gradio as gr

from src import config
from src.inference import get_engine
from src import viz

# Il modello viene caricato all'avvio, non alla prima richiesta: il caricamento
# dei pesi e la compilazione JIT delle librerie audio costano più dell'inferenza,
# e farli pagare al primo utente significherebbe decine di secondi di attesa.
ENGINE = get_engine()


def _warmup():
    """Prima invocazione a vuoto, per pagare subito i costi di inizializzazione."""
    import numpy as np
    silence = np.zeros(int(config.SAMPLE_RATE * config.DURATION), dtype="float32")
    ENGINE.predict(silence, config.SAMPLE_RATE, with_gradcam=True)


_warmup()


# ----------------------------------------------------------------------
# Funzione richiamata dall'interfaccia
# ----------------------------------------------------------------------
def analizza(audio, spiegazione: bool):
    if audio is None:
        raise gr.Error("Registra la voce o carica un file audio prima di avviare l'analisi.")

    sr_in, y = audio                       # Gradio consegna (frequenza, array)
    res = ENGINE.predict(y, sr_in, with_gradcam=spiegazione)

    emozioni = {config.EMOTIONS_IT[e]: p for e, p in res["probabilities"].items()}
    valenza = res["valence"]
    predetta_it = config.EMOTIONS_IT[res["predicted"]]

    fig_wave = viz.waveform_figure(res["waveform"])
    fig_mel = viz.spectrogram_figure(res["mel"])
    fig_cam = viz.gradcam_figure(res["mel"], res["cam"], predetta_it) if spiegazione else None

    t = res["timings"]
    righe = [
        "| Stadio | Tempo |",
        "|---|---:|",
        f"| Pre-elaborazione del segnale | {t['preprocessing_ms']:.1f} ms |",
        f"| Costruzione del tensore | {t['tensor_ms']:.1f} ms |",
        f"| Inferenza della rete | {t['inference_ms']:.1f} ms |",
    ]
    if spiegazione:
        righe.append(f"| Spiegazione (CAM) | {t['spiegazione_ms']:.1f} ms |")
    righe.append(f"| **Totale** | **{t['totale_ms']:.1f} ms** |")
    tempi = "\n".join(righe)

    return emozioni, valenza, fig_wave, fig_mel, fig_cam, tempi


# ----------------------------------------------------------------------
# Interfaccia
# ----------------------------------------------------------------------
DESCRIZIONE = """
Il sistema analizza **come** viene detta una frase, non cosa viene detto: l'emozione
nel parlato è portata dalla prosodia — melodia, intensità, ritmo, dinamica spettrale —
e non dal contenuto lessicale.

La registrazione viene riportata a una durata fissa di 3 secondi, convertita in uno
spettrogramma log-Mel e classificata da una rete convoluzionale su otto emozioni.
"""

NOTA = """
### Come leggere i risultati

**Probabilità.** Il sistema non restituisce solo l'etichetta vincente ma l'intera
distribuzione: una predizione incerta ripartita tra due emozioni vicine è
un'informazione, non un difetto.

**Valenza.** Le otto emozioni sono proiettate sull'asse piacevole/spiacevole del
modello circomplesso di Russell, sommando le probabilità di ciascun gruppo.

**Grad-CAM.** Le zone in ambra sono le regioni del piano tempo-frequenza che hanno
pesato di più sulla decisione. Tipicamente le emozioni ad alta attivazione
attivano le bande medio-alte, quelle a bassa attivazione le bande basse.

**Tempi.** Nella formulazione generale la spiegazione richiede una propagazione
all'indietro attraverso la rete e costa più della predizione stessa. Poiché questa
architettura termina con un global average pooling e un solo strato lineare, la
mappa si ottiene invece dalla medesima passata in avanti, a costo trascurabile.
"""

with gr.Blocks(title="Riconoscimento delle emozioni dal parlato",
               theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Riconoscimento delle emozioni dal parlato")
    gr.Markdown(DESCRIZIONE)

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                label="Registrazione")
            spiegazione = gr.Checkbox(value=True, label="Calcola la spiegazione visiva (Grad-CAM)")
            bottone = gr.Button("Analizza", variant="primary")
            out_emozioni = gr.Label(num_top_classes=8, label="Emozione")
            out_valenza = gr.Label(num_top_classes=3, label="Valenza")

        with gr.Column(scale=2):
            out_wave = gr.Plot(label="Segnale")
            out_mel = gr.Plot(label="Rappresentazione tempo-frequenza")
            out_cam = gr.Plot(label="Spiegazione")
            out_tempi = gr.Markdown(label="Tempi di risposta")

    with gr.Accordion("Note sul funzionamento", open=False):
        gr.Markdown(NOTA)

    gr.Markdown(
        f"<sub>Rete convoluzionale, {ENGINE.n_params:,} parametri. "
        "Addestramento e valutazione con separazione rigorosa dei parlanti: nessuna "
        "voce compare in più di una partizione.</sub>".replace(",", ".")
    )

    bottone.click(
        fn=analizza,
        inputs=[audio_in, spiegazione],
        outputs=[out_emozioni, out_valenza, out_wave, out_mel, out_cam, out_tempi],
    )


if __name__ == "__main__":
    demo.launch()
