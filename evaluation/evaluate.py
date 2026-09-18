"""
Valutazione sperimentale del classificatore.

Tre livelli di analisi:

1. **Riconoscimento a otto classi.** Accuratezza, macro-F1 e balanced accuracy,
   confrontate con i riferimenti che un sistema deve battere per dimostrare di
   aver imparato qualcosa (classe maggioritaria, predizione casuale stratificata,
   caso uniforme) e con un classificatore classico su descrittori MFCC.

2. **Riconoscimento della valenza (tre classi).** Le otto emozioni sono proiettate
   sull'asse piacevole/spiacevole. Due strategie di proiezione, sperimentalmente
   distinguibili, vengono confrontate con un test di significatività:
     - *decisione poi proiezione*: si sceglie l'emozione più probabile e la si mappa;
     - *proiezione poi decisione*: si sommano le probabilità di ciascun gruppo e si
       sceglie il gruppo più probabile.
   La seconda sfrutta l'intera distribuzione: una predizione incerta ripartita fra
   due emozioni negative diverse indica comunque valenza negativa.

3. **Analisi degli errori.** Quanti errori di riconoscimento dell'emozione vengono
   *assorbiti* dal raggruppamento in valenze (confusioni interne a un gruppo) e
   quanti invece *sopravvivono* (confusioni che attraversano il confine fra gruppi).

Uso:  python -m evaluation.evaluate data/export_test.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import binomtest

from src import config

RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000


# ----------------------------------------------------------------------
# Metriche
# ----------------------------------------------------------------------
def metriche(y_true, y_pred, labels=None) -> dict:
    return {
        "accuratezza": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
    }


def bootstrap_ic(y_true, y_pred, n=N_BOOTSTRAP, alpha=0.05) -> dict:
    """
    Intervallo di confidenza percentile per accuratezza e macro-F1.

    Il test set conta 240 clip: una differenza di due punti percentuali equivale a
    cinque clip. L'intervallo quantifica quanto di ciò che osserviamo è segnale e
    quanto è la variabilità del campione.
    """
    n_obs = len(y_true)
    acc, f1 = [], []
    for _ in range(n):
        idx = RNG.integers(0, n_obs, n_obs)
        if len(np.unique(y_true[idx])) < 2:
            continue
        acc.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "accuratezza_ic95": [round(float(np.percentile(acc, lo)), 4),
                             round(float(np.percentile(acc, hi)), 4)],
        "macro_f1_ic95": [round(float(np.percentile(f1, lo)), 4),
                          round(float(np.percentile(f1, hi)), 4)],
    }


def mcnemar_esatto(y_true, pred_a, pred_b) -> dict:
    """
    Test di McNemar in versione esatta.

    Confronta due sistemi sugli *stessi* campioni. Contano solo i casi in cui i due
    sistemi divergono: se A azzecca dove B sbaglia tante volte quante il contrario,
    la differenza complessiva è compatibile con il caso.
    """
    a_ok = (pred_a == y_true)
    b_ok = (pred_b == y_true)
    n01 = int(np.sum(a_ok & ~b_ok))     # solo A corretto
    n10 = int(np.sum(~a_ok & b_ok))     # solo B corretto
    discordanti = n01 + n10
    p = float(binomtest(n01, discordanti, 0.5).pvalue) if discordanti else 1.0
    return {
        "solo_A_corretto": n01,
        "solo_B_corretto": n10,
        "entrambi_corretti": int(np.sum(a_ok & b_ok)),
        "entrambi_errati": int(np.sum(~a_ok & ~b_ok)),
        "p_value": round(p, 4),
        "significativo_al_5pc": bool(p < 0.05),
    }


# ----------------------------------------------------------------------
# Proiezione sull'asse della valenza
# ----------------------------------------------------------------------
def indici_valenza(emozioni):
    """Per ogni classe di valenza, gli indici delle emozioni che vi appartengono."""
    return {v: [i for i, e in enumerate(emozioni) if config.VALENCE_MAP[e] == v]
            for v in config.VALENCE_CLASSES}


def proietta_etichette(y, emozioni, gruppi):
    """Etichette di emozione -> etichette di valenza."""
    ordine = {v: k for k, v in enumerate(config.VALENCE_CLASSES)}
    return np.array([ordine[config.VALENCE_MAP[emozioni[i]]] for i in y])


def decisione_poi_proiezione(probs, emozioni):
    return proietta_etichette(probs.argmax(1), emozioni, None)


def proiezione_poi_decisione(probs, emozioni):
    gruppi = indici_valenza(emozioni)
    somme = np.stack([probs[:, gruppi[v]].sum(1) for v in config.VALENCE_CLASSES], axis=1)
    return somme.argmax(1)


# ----------------------------------------------------------------------
# Analisi degli errori
# ----------------------------------------------------------------------
def analisi_errori(y_true, y_pred, emozioni) -> dict:
    """Errori di emozione assorbiti dal raggruppamento in valenze, contro errori che sopravvivono."""
    errati = np.where(y_pred != y_true)[0]
    assorbiti, sopravvissuti = [], []
    for i in errati:
        stessa = config.VALENCE_MAP[emozioni[y_true[i]]] == config.VALENCE_MAP[emozioni[y_pred[i]]]
        (assorbiti if stessa else sopravvissuti).append(i)

    coppie = {}
    for i in assorbiti:
        k = f"{emozioni[y_true[i]]} -> {emozioni[y_pred[i]]}"
        coppie[k] = coppie.get(k, 0) + 1

    return {
        "clip_totali": int(len(y_true)),
        "errori_a_8_classi": int(len(errati)),
        "errori_assorbiti_dal_raggruppamento": int(len(assorbiti)),
        "errori_sopravvissuti": int(len(sopravvissuti)),
        "quota_assorbita": round(len(assorbiti) / max(len(errati), 1), 4),
        "confusioni_assorbite_piu_frequenti": dict(
            sorted(coppie.items(), key=lambda kv: -kv[1])[:8]),
    }


# ----------------------------------------------------------------------
def main(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    probs = d["probs"]
    y_true = d["y_true"]
    emozioni = [str(e) for e in d["emotions"]]

    y_pred = probs.argmax(1)
    out = {"protocollo": "separazione rigorosa dei parlanti (16 addestramento / 4 validazione / 4 test)",
           "clip_di_test": int(len(y_true))}

    # ---------------- otto classi ----------------
    dummy_maj = DummyClassifier(strategy="most_frequent").fit(d["X_vec_train"], d["y_train"])
    dummy_str = DummyClassifier(strategy="stratified", random_state=42).fit(d["X_vec_train"], d["y_train"])
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(d["X_vec_train"], d["y_train"])

    pred_maj8 = dummy_maj.predict(d["X_vec_test"])
    pred_str8 = dummy_str.predict(d["X_vec_test"])
    pred_svm8 = svm.predict(d["X_vec_test"])

    out["otto_classi"] = {
        "CNN": {**metriche(y_true, y_pred), **bootstrap_ic(y_true, y_pred)},
        "SVM_su_MFCC": metriche(y_true, pred_svm8),
        "riferimento_classe_maggioritaria": metriche(y_true, pred_maj8),
        "riferimento_casuale_stratificato": metriche(y_true, pred_str8),
        "caso_uniforme": round(1 / len(emozioni), 4),
        "report_per_classe": classification_report(
            y_true, y_pred, target_names=emozioni, zero_division=0, output_dict=True),
        "matrice_confusione": confusion_matrix(y_true, y_pred).tolist(),
    }

    # ---------------- tre classi (valenza) ----------------
    yv_true = proietta_etichette(y_true, emozioni, None)
    yv_a = decisione_poi_proiezione(probs, emozioni)
    yv_b = proiezione_poi_decisione(probs, emozioni)
    yv_svm = proietta_etichette(pred_svm8, emozioni, None)
    yv_maj = np.full_like(yv_true, np.bincount(proietta_etichette(d["y_train"], emozioni, None)).argmax())

    dist = {v: int(np.sum(yv_true == i)) for i, v in enumerate(config.VALENCE_CLASSES)}
    out["valenza"] = {
        "distribuzione_test": dist,
        "quota_classe_maggioritaria": round(max(dist.values()) / len(yv_true), 4),
        "A_decisione_poi_proiezione": {**metriche(yv_true, yv_a), **bootstrap_ic(yv_true, yv_a)},
        "B_proiezione_poi_decisione": {**metriche(yv_true, yv_b), **bootstrap_ic(yv_true, yv_b)},
        "SVM_su_MFCC": metriche(yv_true, yv_svm),
        "riferimento_classe_maggioritaria": metriche(yv_true, yv_maj),
        "mcnemar_A_vs_B": mcnemar_esatto(yv_true, yv_a, yv_b),
        "report_per_classe_B": classification_report(
            yv_true, yv_b, target_names=config.VALENCE_CLASSES,
            zero_division=0, output_dict=True),
        "matrice_confusione_B": confusion_matrix(yv_true, yv_b).tolist(),
    }

    # ---------------- errori ----------------
    out["analisi_errori"] = analisi_errori(y_true, y_pred, emozioni)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.RESULTS_DIR / "valutazione.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "otto_classi"},
                     indent=2, ensure_ascii=False))
    print(f"\nRisultati completi in {dest}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/export_test.npz")
