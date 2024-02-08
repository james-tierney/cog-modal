import base64
import datetime
import os
import re
import subprocess
import time
from typing import List

import requests
import torch

from cog import BasePredictor, BaseModel, Input, Path

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from modal import Stub 

stub = Stub("cog model")

class Output(BaseModel):
    segments: list
    language: str = None
    num_speakers: int = None

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("model loaded")
        model_name = "large-v3"
        self.model = WhisperModel(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16"
        )
        self.diarization_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(torch.device("cuda"))

    @stub.function
    def predict(
        self,
        file_string=None,
        file_url= "https://replicate.delivery/pbxt/JcL0ttZLlbchC0tL9ZtB20phzeXCSuMm0EJNdLYElgILoZci/AI%20should%20be%20open-sourced.mp3",
        file=None,
        group_segments=True,
        num_speakers=2,
        language="English",
        prompt="Mark and Lex talking about AI.",
        offset_seconds=0
    ):
        """Run a single prediction on the model"""
        try:
            # Generate a temporary filename
            temp_wav_filename = f"temp-{time.time_ns()}.wav"

            if file is not None:
                subprocess.run([
                    'ffmpeg', '-i', file, '-ar', '16000', '-ac', '1', '-c:a',
                    'pcm_s16le', temp_wav_filename
                ])
            elif file_url is not None:
                response = requests.get(file_url)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, 'wb') as file:
                    file.write(response.content)
                subprocess.run([
                    'ffmpeg', '-i', temp_audio_filename, '-ar', '16000', '-ac',
                    '1', '-c:a', 'pcm_s16le', temp_wav_filename
                ])
                os.remove(temp_audio_filename)
            elif file_string is not None:
                audio_data = base64.b64decode(file_string.split(',')[1] if ',' in file_string else file_string)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, 'wb') as f:
                    f.write(audio_data)
                subprocess.run([
                    'ffmpeg', '-i', temp_audio_filename, '-ar', '16000', '-ac',
                    '1', '-c:a', 'pcm_s16le', temp_wav_filename
                ])
                os.remove(temp_audio_filename)

            segments, detected_num_speakers, detected_language = self.speech_to_text(temp_wav_filename, num_speakers, prompt, offset_seconds, group_segments, language, word_timestamps=True)

            return Output(segments=segments, language=detected_language, num_speakers=detected_num_speakers)

        except Exception as e:
            raise RuntimeError("Error running inference with the model", e)

        finally:
            # Clean up
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def speech_to_text(
        self,
        audio_file_wav,
        num_speakers=None,
        prompt="Your, vocabulary, here. Use punctuation for best accuracy.",
        offset_seconds=0,
        group_segments=True,
        language=None,
        word_timestamps=True
    ):
        time_start = time.time()

        # Transcribe audio
        print("Starting transcription")
        options = dict(vad_filter=True, initial_prompt=prompt, word_timestamps=word_timestamps, language=language)
        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [{
            'start': float(s.start + offset_seconds),
            'end': float(s.end + offset_seconds),
            'text': s.text,
            'words': [{
                'start': float(w.start + offset_seconds),
                'end': float(w.end + offset_seconds),
                'word': w.word
            } for w in s.words]
        } for s in segments]

        time_transcribing_end = time.time()
        print(f"Finished transcription, took {time_transcribing_end - time_start:.5} seconds")

        diarization = self.diarization_model(audio_file_wav, num_speakers=num_speakers)
        time_diarization_end = time.time()
        print(f"Finished diarization, took {time_diarization_end - time_transcribing_end:.5} seconds")

        # Initialize variables to keep track of the current position in both lists
        margin = 0.1  # 0.1 seconds margin
        final_segments = []

        diarization_list = list(diarization.itertracks(yield_label=True))
        unique_speakers = {speaker for _, _, speaker in diarization.itertracks(yield_label=True)}
        detected_num_speakers = len(unique_speakers)

        speaker_idx = 0
        n_speakers = len(diarization_list)

        # Iterate over each segment
        for segment in segments:
            segment_start = segment['start'] + offset_seconds
            segment_end = segment['end'] + offset_seconds
            segment_text = []
            segment_words = []

            # Iterate over each word in the segment
            for word in segment['words']:
                word_start = word['start'] + offset_seconds - margin
                word_end = word['end'] + offset_seconds + margin

                while speaker_idx < n_speakers:
                    turn, _, speaker = diarization_list[speaker_idx]

                    if turn.start <= word_end and turn.end >= word_start:
                        # Add word without modifications
                        segment_text.append(word['word'])
                        # Strip here for individual word storage
                        word['word'] = word['word'].strip()
                        segment_words.append(word)

                        if turn.end <= word_end:
                            speaker_idx += 1
                        break
                    elif turn.end < word_start:
                        speaker_idx += 1
                    else:
                        break

            if segment_text:
                combined_text = ''.join(segment_text)
                cleaned_text = re.sub('  ', ' ', combined_text).strip()
                new_segment = {
                    'start': segment_start - offset_seconds,
                    'end': segment_end - offset_seconds,
                    'speaker': speaker,
                    'text': cleaned_text,
                    'words': segment_words
                }
                final_segments.append(new_segment)

        time_merging_end = time.time()
        print(f"Finished merging, took {time_merging_end - time_diarization_end:.5} seconds")
        segments = final_segments
        output = []  # Initialize an empty list for the output

        # Initialize the first group with the first segment
        current_group = {
            'start': str(segments[0]["start"]),
            'end': str(segments[0]["end"]),
            'speaker': segments[0]["speaker"],
            'text': segments[0]["text"],
            'words': segments[0]["words"]
        }

        for i in range(1, len(segments)):
            time_gap = segments[i]["start"] - segments[i - 1]["end"]
            if segments[i]["speaker"] == segments[i - 1]["speaker"] and time_gap <= 2 and group_segments:
                current_group["end"] = str(segments[i]["end"])
                current_group["text"] += " " + segments[i]["text"]
                current_group["words"] += segments[i]["words"]
            else:
                output.append(current_group)
                current_group = {
                    'start': str(segments[i]["start"]),
                    'end': str(segments[i]["end"]),
                    'speaker': segments[i]["speaker"],
                    'text': segments[i]["text"],
                    'words': segments[i]["words"]
                }

        output.append(current_group)
        time_cleaning_end = time.time()
        print(f"Finished cleaning, took {time_cleaning_end - time_merging_end:.5} seconds")
        time_end = time.time()
        time_diff = time_end - time_start
        print(f"Processing time: {time_diff:.5} seconds")

        return output, detected_num_speakers, transcript_info.language


@stub.local_entrypoint()
def run_model(file_string: str = None, file_url: str = None, file_path: str = None):
    # Instantiate the Predictor class
    predictor = Predictor()

    # Call the setup method to load the model into memory
    predictor.setup()

    # Call the predict method with the input data
    if file_string:
        output = predictor.predict(file_string=file_string)
    elif file_url:
        output = predictor.predict(file_url=file_url)
    elif file_path:
        output = predictor.predict(file=file_path)
    else:
        raise ValueError("Please provide either file string, file URL, or file path.")

    # Return the prediction output
    return output

if __name__ == "__main__":

    file_path = "https://replicate.delivery/pbxt/JcL0ttZLlbchC0tL9ZtB20phzeXCSuMm0EJNdLYElgILoZci/AI%20should%20be%20open-sourced.mp3"

    # Call the remote entry point function with the input data
    result = run_model(file_path=file_path)

    # Print the result
    print(result)
