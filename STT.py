import pandas as pd
import speech_recognition as sr

class Speech_To_Text:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def read_audio(self, audio_file):
        # Use the audio file as the audio source
        with sr.AudioFile(audio_file) as source:
            self.audio_data = self.recognizer.record(source)  # read the entire audio file

    def return_text(self):
        # Recognize speech using Google Web Speech API
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            return self.recognizer.recognize_google(self.audio_data, language="ko-KR")
        except sr.UnknownValueError:
            return "Google Web Speech could not understand audio"
        except sr.RequestError as e:
            return "Could not request results from Google Web Speech service; {0}".format(e)


def speech_to_text(wav_id, files_path):
    # # Initialize recognizer
    STT = Speech_To_Text()

    texts =[]
    for file in wav_id:
        STT.read_audio(files_path + file)
        text = STT.return_text()
        texts.append(text)

    return texts
