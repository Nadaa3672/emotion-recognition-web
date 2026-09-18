"""
Grad-CAM: localizzazione delle regioni dello spettrogramma che determinano la decisione.

L'idea: si prendono le mappe di attivazione dell'ultimo blocco convoluzionale e
si pesano con i gradienti della classe predetta rispetto a quelle attivazioni.
Un canale i cui valori, se aumentassero, farebbero crescere molto il punteggio
della classe riceve peso alto. La somma pesata, passata per una ReLU e riportata
alla dimensione dello spettrogramma, indica *dove* nel piano tempo-frequenza la
rete sta "ascoltando".

Nel contesto di un servizio web questo trasforma una predizione opaca in una
risposta ispezionabile dall'utente.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
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
        """x: tensore (1, C, H, W). Restituisce (mappa HxW normalizzata in [0,1], indice di classe)."""
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1).item())
        out[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), class_idx


def last_conv_block(model):
    """Ultimo blocco convoluzionale, immediatamente prima del global pooling."""
    return model.features[2]
