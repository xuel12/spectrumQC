import json
import numpy as np


class Spectrum:
    def __init__(self, signal, idx=None, label=None):
        self.idx = idx
        self.label = label

        if type(signal) != np.ndarray:
            raise TypeError(
                f"Signal should be initialized with a numpy array, and not a {type(signal)}"
            )
        if not (len(signal.shape) == 2 and (signal.shape[1] == 2)):
            raise TypeError(
                f"Signal should be initialized with a 2D, 2-column numpy array. Instead it got a numpy array of shape {signal.shape}"
            )
        self._t = signal[:, 0]
        self._s = signal[:, 1]

    # Only getters, no setters.
    @property
    def s(self):
        return self._s

    @property
    def t(self):
        return self._t

    @classmethod
    def from_json(cls, json_data):
        # json_dict = json.loads(json_data)
        json_dict = json_data

        spectrum = cls(
            signal=np.array([json_dict["t"], json_dict["s"]]).T,
            idx=json_dict["idx"],
            label=json_dict["label"],
        )

        return spectrum

    def to_json(self):
        try:
            t = self._t.tolist()
            s = self._s.tolist()
        except AttributeError:
            t = None
            s = None

        json_dict = {"idx": self.idx, "label": self.label, "t": t, "s": s}

        return json_dict
