"""
Spiegazione della decisione per localizzazione sul piano tempo-frequenza.

Grad-CAM pesa le mappe di attivazione dell'ultimo blocco convoluzionale con i
gradienti della classe predetta rispetto a quelle attivazioni: un canale i cui
valori, crescendo, farebbero salire molto il punteggio della classe riceve peso
alto. La somma pesata, passata per una ReLU e riportata alla dimensione dello
spettrogramma, indica *dove* la rete sta ascoltando.

Due implementazioni equivalenti, e il motivo per cui lo sono.

`GradCAM` è la formulazione generale: richiede una propagazione all'indietro
attraverso la rete, e in un servizio interattivo è lo stadio più costoso
dell'intera catena.

`FastCAM` sfrutta invece la struttura specifica di questa architettura. Dopo
l'ultimo blocco convoluzionale la rete applica un global average pooling seguito
da un unico strato lineare, quindi il logit della classe c vale

    z_c = somma_k [ w_ck * media_ij( A_k[i,j] ) ] + b_c

e il gradiente rispetto a ogni attivazione è costante:

    d z_c / d A_k[i,j] = w_ck / (h*w)

Il peso Grad-CAM del canale k, che è la media di quel gradiente, coincide dunque
con il peso w_ck dello strato lineare, a meno di un fattore costante che la
normalizzazione finale in [0,1] elimina. La mappa si ottiene perciò **senza
propagazione all'indietro**, riusando le attivazioni della stessa passata in
avanti che produce la predizione.

Questa equivalenza è esatta per architetture con global average pooling e un solo
strato lineare terminale; è il caso della rete qui impiegata.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _normalizza(cam: np.ndarray) -> np.ndarray:
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


class GradCAM:
    """Formulazione generale, con propagazione all'indietro. Usata come riferimento."""

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1).item())
        out[0, class_idx].backward()

        pesi = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((pesi * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        return _normalizza(cam.squeeze().cpu().numpy()), class_idx


class FastCAM:
    """
    Stessa mappa, in una sola passata in avanti.

    Le attivazioni dell'ultimo blocco convoluzionale vengono catturate da un hook
    durante la passata che calcola la predizione; i pesi dei canali si leggono
    direttamente dallo strato lineare finale.
    """

    def __init__(self, model, target_layer, linear_layer):
        self.model = model
        self.acts = None
        self.W = linear_layer.weight.detach()          # (n_classi, n_canali)
        target_layer.register_forward_hook(self._fwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    @torch.no_grad()
    def __call__(self, x, class_idx=None, logits=None):
        """
        `logits` evita una seconda passata quando il chiamante ha già calcolato la
        predizione: le attivazioni catturate dall'hook sono quelle di quella stessa
        passata.
        """
        if logits is None:
            logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        pesi = self.W[class_idx].view(1, -1, 1, 1)
        cam = F.relu((pesi * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        return _normalizza(cam.squeeze().cpu().numpy()), class_idx


def last_conv_block(model):
    """Ultimo blocco convoluzionale, immediatamente prima del global pooling."""
    return model.features[2]


def final_linear(model):
    """Strato lineare terminale, i cui pesi fungono da pesi dei canali."""
    return model.classifier[-1]
