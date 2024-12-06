import soundfile as sf

import inference

class Attack():
    def __init__(self, temp_path, watermarker_handler):
        '''
        initializes everything and handle saving the original audio file in .wav format
        Diff-HierVC only works with .wav files
        param: temp_path, str, where to store the .wav file
        param: watermarker_handler, WatermarkerHandler, adds watermark before attack and detects afterwards
            it will already have .samples and .sr set
        '''
        self.watermarker_handler = watermarker_handler
        if self.watermarker_handler.state == "initialized":  
            # just in case the watermark was already added
            self.watermarker_handler.add_watermark() 
        
        self.audio_file_path = temp_path
        sf.write(temp_path, self.watermarker_handler.samples, self.watermarker_handler.sr)
        self.attacked_file = None  # this is set by self.attack
        
        
    def attack(self):
        '''
        run the attack and sets the attacked_file (str) attribute
        '''
        pass

    def attack_results(self):
        '''
        return detection score of watermark after attack
        returns: number from 0 to 1 representing confidence whether or not watermark is there
        '''
        attacked_samples, sr = sf.read(self.attacked_file)
        self.watermarker_handler.samples = attacked_samples
        return self.watermarker_handler.detect_watermark()


# this converts the input to several intermediate speakers and then back
DEFAULT_INTERMEDIATE_SPEAKERS = ["hello.wav", "sample_taiwanese.wav"]
class MultipleConversionAttack(Attack):
    def __init__(self, temp_path, watermarker_handler, intermediate_speakers=DEFAULT_INTERMEDIATE_SPEAKERS):
        '''
        initializes a multiple conversion attack
        param: intermediate_speakers, array of str with audio files of intermediate speakers
        '''
        super().__init__(temp_path, watermarker_handler)
        self.intermediate_speakers = intermediate_speakers

    def attack(self):
        self.attacked_file = self.audio_file_path  # attacked file will accumulate speaker conversions
        for intermediate_speaker in self.intermediate_speakers:
            self.attacked_file = inference.main(self.attacked_file, intermediate_speaker)
        # convert back to original speaker
        self.attacked_file = inference.main(self.attacked_file, self.audio_file_path)

# this converts the input to the same speaker
class SelfConversionAttack(MultipleConversionAttack):
    def __init__(self, temp_path, watermarker_handler, intermediate_speakers=[]):
        '''
        there are no intermediate speakers in self conversion attack
        '''
        super().__init__(temp_path, watermarker_handler, intermediate_speakers)

to_try = [SelfConversionAttack, MultipleConversionAttack]