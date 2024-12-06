import os
from random import randint
import soundfile as sf

import handle_watermark
import attacks

OUTPUT_FILE = "data.csv"
HEADER = "filepath,len_seconds,audioseal,wavmark,combined_audioseal,combined_wavmark,self_conversion_attack_audioseal,self_conversion_attack_wavmark,multiple_conversion_attack_audioseal,multiple_conversion_attack_wavmark"
# Header goes file info, baseline control detection, combined detection, attacked detection for each watermark
AUDIO_DIR = "../librispeech"
OUTPUT_ROWS = 2  # 20k is what audiomarkbench used
AUDIO_FILE_ENDINGS = [".flac", ".wav", ".mp3"]
TEMP_ATTACKED_FILE_PATH = "being_attacked.wav"

def is_audio_file(filename):
    '''
    param: filename, str
    returns: bool, whether or not this is an audio file
    '''
    for audio_file_ending in AUDIO_FILE_ENDINGS:
        if filename.endswith(audio_file_ending):
            return True
    return False

def main():
    '''
    handles building the dataset and calling supporting functions
    '''
    # get all .wav files and put them in one list
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(AUDIO_DIR):
        for filename in filenames:
            if is_audio_file(filename):
                audio_files.append(os.path.join(dirpath, filename))
    
    if len(audio_files) < OUTPUT_ROWS:
        print(f"Error! There are {len(audio_files)} audio files in {AUDIO_DIR}, but {OUTPUT_ROWS} are needed.")
        return

    for row_num in range(OUTPUT_ROWS):
        print("now processing audio file number", row_num + 1, "of", OUTPUT_ROWS)
        row_str_arr = []  # will be joined into a str at the end of the iter
        # sample an audio file to run tests on and remove from audio_files
        rand_ind = randint(0, len(audio_files) - 1)
        curr_audio_file_name = audio_files[rand_ind]
        row_str_arr.append(curr_audio_file_name)
        del audio_files[rand_ind]

        # get the length of the audio file in seconds
        # Reading an audio file
        curr_samples, curr_sr = sf.read(curr_audio_file_name)    
        seconds = len(curr_samples) / curr_sr
        row_str_arr.append(str(seconds))
        print("file name", curr_audio_file_name, "has duration", seconds, "seconds")

        # apply each watermark to test, and detect it
        print("watermark detection baseline test for", row_num, "of", OUTPUT_ROWS)
        for watermark_handler in handle_watermark.watermarks_to_test:
            curr_watermark_handler = watermark_handler(curr_samples, curr_sr)
            curr_watermark_handler.add_watermark()
            detection_score = curr_watermark_handler.detect_watermark()
            row_str_arr.append(str(detection_score))

        # combine the watermarks
        print("watermark interference detection test for", row_num, "of", OUTPUT_ROWS)
        for first_watermarker_ind in range(len(handle_watermark.watermarks_to_test)):
            # pick one watermark to add first
            first_watermarker_class  = handle_watermark.watermarks_to_test[first_watermarker_ind]
            first_watermarker = first_watermarker_class(curr_samples, curr_sr)
            first_watermarker.add_watermark()
            combined_samples = first_watermarker.samples  # this accumulates the watermarks
            for i in range(len(handle_watermark.watermarks_to_test)):
                # add the other watermarks on top
                if i == first_watermarker_ind:
                    continue
                interference_watermarker_class = handle_watermark.watermarks_to_test[i]
                interference_watermarker = interference_watermarker_class(combined_samples, curr_sr)
                interference_watermarker.add_watermark()
                combined_samples = interference_watermarker.samples

            # now see if first watermark can be detected
            first_watermarker.samples = combined_samples
            detection_score = first_watermarker.detect_watermark()
            row_str_arr.append(str(detection_score))

        # run through each attack for each watermarker
        print("running attacks for", row_num, "of", OUTPUT_ROWS)
        for watermark_handler in handle_watermark.watermarks_to_test:
            for attack in attacks.to_try:
                curr_attack = attack(TEMP_ATTACKED_FILE_PATH, watermark_handler(curr_samples, curr_sr))
                curr_attack.attack()
                detection_score = curr_attack.attack_results()
                row_str_arr.append(str(detection_score))

        # write the result to the output file
        print("writing results for", row_num, "of", OUTPUT_ROWS)
        with open(OUTPUT_FILE, "a") as output_file:
            row_str_no_newline = ",".join(row_str_arr)
            row_str = "".join([row_str_no_newline, "\n"])
            output_file.write(row_str)

main()