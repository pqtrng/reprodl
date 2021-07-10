import io
import json
import logging
import os

import hydra
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


from model import AudioNet


class Handler:
    def __init__(self) -> None:
        self.model = None
        self.mapping = None
        self.device = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx) -> None:
        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        model_dir = properties.get("model_dir")
        model_pt_path = os.path.join(model_dir, "model.path")
        with open("index_to_name.json") as json_file:
            self.mapping = json.load(json_file)

        init_cfg = hydra.initialize(config_path="./", job_name="test_app")
        self.cfg = hydra.compose(config_name="default")

        self.model = AudioNet(hparams=self.cfg.model)

        state_dict = torch.load(f=model_pt_path, map_location=self.device)
        self.model.load_state_dict(state_dict=state_dict)
        self.model.to(self.device)

        logger.debug(f"Model file {model_pt_path} loaded successfully.")
        self.initialized = True

    def pre_process(self, data):
        audio = data[0].get("data")
        if audio is None:
            audio = data[0].get("body")

        resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=8000)
        melspec = torchaudio.transforms.MelSpectrogram(sample_rate=8000)
        db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        wav, samplerate = sf.read(io.BytesIO(audio))

        wav = torch.FloatTensor(wav)

        x_b = resample(wav.unsqueeze(0))
        x_b = melspec(x_b)
        x_b = db(x_b)

        return x_b.unsqueeze(0)

    def inference(self, audio) -> list:
        self.model.eval()
        y_pred = self.model.forward(audio)
        predicted_idx = y_pred.argmax(-1).item()

        return [str(predicted_idx)]

    def post_process(self, inference_output):
        res = []
        for pred in inference_output:
            label = self.mapping[str(pred)][1]
            label = "\n\t" + label + "\n\n"
            res.append(label)
        return res


_service = Handler()


def handle(data, context):

    if not _service.initialized:
        _service.initialize(ctx=context)

    if data is None:
        return None

    data = _service.pre_process(data=data)
    data = _service.inference(audio=data)
    data = _service.post_process(inference_output=data)

    return data
