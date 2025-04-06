from models import SMETimes_Llama, SMETimes_Gpt2, SMETimes_Opt_1b

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SMETimes_Llama': SMETimes_Llama,
            'SMETimes_Gpt2': SMETimes_Gpt2,
            'SMETimes_Opt_1b': SMETimes_Opt_1b
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
