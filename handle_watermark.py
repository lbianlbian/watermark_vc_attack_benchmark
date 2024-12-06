class WatermarkHandler():
    def __init__(self, samples, sr):
        '''
        param: samples, np.array, is an array of the samples of the audio
        param: sr, int, is the sample rate of the audio
        '''
        self.samples = samples
        self.sr = sr
        self.state = "initialized"  # either initialized, watermarked, or attacked
    
    def add_watermark(self):
        '''
        adds the watermark to self.audio_file
        '''
        self.state = "watermarked"
    
    def detect_watermark(self):
        '''
        should be called after self.add(). returns detection score for self.audio_file
        returns: number in between 0 and 1
        '''
        pass

import torch
from audioseal import AudioSeal

torch.set_default_dtype(torch.float64)  # without this line, watermark insertion throws expected double got float error no matter wahat input type is
audioseal_watermarker = AudioSeal.load_generator("audioseal_wm_16bits")
audioseal_detector = AudioSeal.load_detector("audioseal_detector_16bits")
BATCH_DIM = 1
CHANNEL_DIM = 1

class AudioSealHandler(WatermarkHandler):
    def reshape(self):
        tensor = torch.from_numpy(self.samples)
        double_tensor = tensor.double()
        self.reshaped = double_tensor.view(BATCH_DIM, CHANNEL_DIM, len(self.samples))

    def add_watermark(self):
        super().add_watermark()
        self.reshape()
        watermark = audioseal_watermarker.get_watermark(self.reshaped, sample_rate=self.sr)
        self.reshaped += watermark
        self.samples = self.reshaped[0][0].detach().numpy()
        
    def detect_watermark(self):
        self.reshape()
        result, message = audioseal_detector.detect_watermark(self.reshaped, self.sr)
        return result

import wavmark
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)  # may have been set to float64/double elsewhere, which breaks wavmark
model = wavmark.load_model().to(device)

class WavmarkHandler(WatermarkHandler):
    def __init__(self, samples, sr):
        super().__init__(samples, sr)
        self.payload = np.random.choice([0, 1], size=16)

    def add_watermark(self):
        super().add_watermark()
        self.samples, _ = wavmark.encode_watermark(model, self.samples, self.payload, show_progress=True)
        
    def detect_watermark(self):
        payload_decoded, _ = wavmark.decode_watermark(model, self.samples, show_progress=True)
        BER = (self.payload != payload_decoded).mean() * 100  # byte err rate, in (0, 1), 0.5 = random guessing
        return 1 - BER  # I want a higher number = yes this is watermarked. 

watermarks_to_test = [AudioSealHandler, WavmarkHandler]