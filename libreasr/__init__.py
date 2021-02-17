"""
LibreASR source code
"""
import gc

from libreasr.lib.defaults import DEFAULT_CONFIG_PATH
from libreasr.lib.inference.utils import load_config
from libreasr.lib.instance import LibreASRTraining, LibreASRInference


class LibreASR:
    def __init__(self, lang, config_path=DEFAULT_CONFIG_PATH):
        """
        Create a new LibreASR instace for a specific language
        """
        self.lang = lang
        self.config_path = config_path
        self.conf = load_config(config_path, lang)
        self.inst = None
        self.mode = None

    def _collect_garbage(self, ok, new_mode):
        if ok and self.mode != new_mode:
            if self.inst is not None:
                self.inst = None
                self.mode = None
                gc.collect()
                print("[LibreASR] garbage collected")
            return True
        return False

    def load_training(self, do_gc=True):
        m = "training"
        if self._collect_garbage(do_gc, m):
            self.inst = LibreASRTraining(self.lang, self.config_path)
            self.mode = m

    def load_inference(self, do_gc=True):
        m = "inference"
        if self._collect_garbage(do_gc, m):
            self.inst = LibreASRInference(self.lang, self.config_path)
            self.mode = m

    def transcribe(self, sth, batch=True, load=True, **kwargs):
        """
        Transcribe files or tensors
        """
        if load:
            self.load_inference()
        return self.inst.transcribe(sth, batch=batch, **kwargs)

    def stream(self, sth, load=True, **kwargs):
        """
        Transcribe stuff in a stream
        """
        if load:
            self.load_inference()
        return self.inst.stream(sth, **kwargs)

    def train(self):
        """
        Train a model
        """
        self.load_training()

    def validate(self, pcent=1.0):
        """
        Validate a saved model.
        This uses the `valid` dataset.
        """
        self.load_inference()

    def test(self, pcent=1.0):
        """
        Test a saved model.
        This uses the `test` dataset.
        """
        self.load_inference()

    def serve(self, port):
        """
        Run the LibreASR API-Server
        """
        pass
