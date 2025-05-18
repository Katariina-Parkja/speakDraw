import tkinter as tk
from tkinter import Label, Message
from PIL import Image, ImageTk
import threading
import numpy as np
import sounddevice as sd
from diffusers import StableDiffusionPipeline
from speechbrain.inference import EncoderClassifier
import torch
import re
import time
import soundfile as sf
import whisper
# from whispercpp import Whisper # use whispercpp if working with limited VRAM

from transformers import MarianMTModel, MarianTokenizer
from espnet2.bin.asr_inference import Speech2Text


def translate(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translated_tokens = translator.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(translated_text)
    return translated_text


def clean_transcription(text):
    cleaned_text = re.sub(r"[(\[].*?[)\]]", "", text)
    return " ".join(cleaned_text.split())


def detect_language(audio):
    out_prob, score, index, text_lab = classifier.classify_file(audio)
    en_idx = classifier.hparams.label_encoder.lab2ind["en: English"]
    et_idx = classifier.hparams.label_encoder.lab2ind["et: Estonian"]
    en_score = out_prob[0][en_idx]
    et_score = out_prob[0][et_idx]
    print(en_score, et_score)
    if en_score > et_score:
        print("English")
        return "en"
    return "et"



class App:
    def __init__(self, root):
        self.root = root
        self.running = True
        self.root.title("SpeakDraw")
        self.root.attributes("-fullscreen", True)
        self.root['bg'] = "black"
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.loading_transcription = False
        self.loading_image = False

        self.center_text = Message(self.root, text="", font=("Arial", 44), fg="white", bg="black", width=root.winfo_width()*0.9, justify="center")
        self.center_text.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.image_label = Label(self.root)  # Image display
        self.center_text.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        self.wake_text_est = Label(self.root, text='Käivita mikrofon ja kirjelda pilti, mida soovid näha.\nVaigista mikrofon uuesti, kui oled rääkimise lõpetanud.', font=("Arial", 24), fg="white", bg="black")
        self.wake_text_est.place(relx=0.5, rely=0.45, anchor=tk.CENTER)
        self.wake_text_eng = Label(self.root, text='Unmute the microphone and describe the image you want to see.\nMute the microphone again once you have finished speaking.', font=("Arial", 24), fg="white", bg="black")
        self.wake_text_eng.place(relx=0.5, rely=0.55, anchor=tk.CENTER)
        self.previous_prompt_est = Label(self.root, font=("Arial", 20), fg="white", bg="black")
        self.previous_prompt_eng = Label(self.root, font=("Arial", 20), fg="white", bg="black")

        self.backend_thread = threading.Thread(target=self.record_command, daemon=True)
        self.backend_thread.start()


    def update_text(self, text: str):
        self.wake_text_est.place_forget()
        self.wake_text_eng.place_forget()
        self.image_label.place_forget()
        self.previous_prompt_est.place_forget()
        self.previous_prompt_eng.place_forget()
        self.center_text.config(text=text)
        self.center_text.place(relx=0.5, rely=0.5, anchor=tk.CENTER)


    def update_listening_labels(self, lang: str, prompt: str):
        self.wake_text_est.place(relx=0.5, rely=0.80, anchor=tk.CENTER)
        self.wake_text_eng.place(relx=0.5, rely=0.88, anchor=tk.CENTER)
        if lang == "en":
            self.previous_prompt_eng.config(text=f"Previous prompt: {prompt}")
            self.previous_prompt_eng.place(relx=0.5, rely=0.72, anchor=tk.CENTER)
        else:
            self.previous_prompt_est.config(text=f"Eelmine sisendtekst: {prompt}")
            self.previous_prompt_est.place(relx=0.5, rely=0.72, anchor=tk.CENTER)

    def display_image(self, image_path):
        img = Image.open(image_path)
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.center_text.place_forget()
        self.image_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)


    def on_close(self):
        self.running = False
        self.backend_thread.join(timeout=1)
        self.root.destroy()
        exit(0)


    def record_command(self):
        command_audio = []
        with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=1024) as stream:
            while True:
                data, _ = stream.read(1024)
                pcm = np.frombuffer(data, dtype=np.int16)

                if len(pcm) == 0:
                    continue

                energy = np.linalg.norm(pcm)

                if len(command_audio) > 0 and energy == 0:
                    self.loading_transcription = True
                    break

                if energy == 0.0:
                    continue

                command_audio.append(data)

        audio_np = np.concatenate(command_audio, axis=0).astype(np.float32) / 32768.0
        sf.write("audio.wav", audio_np, 16000)
        lang = detect_language("audio.wav")
        if lang == "et":
            self.update_text("Kuulamine lõpetatud.\nTöötlen sinu pildikirjeldust...")
        else:
            self.update_text("Recording complete.\nProcessing your image description...")
        print(f"Audio Stats - Min: {audio_np.min()}, Max: {audio_np.max()}, Mean: {audio_np.mean()}")
        self.transcribe_command(audio_np, lang)


    def transcribe_command(self, audio, lang: str):
        start = time.time()
        if lang == "et":
            result, *_ = est_model(audio)
            result = result[0]
        else:
            result = transcriber.transcribe("audio.wav")["text"]
            # result = transcriber.transcribe(audio) # if using whispercpp

        end = time.time()
        print("Transcription time:", end-start, "s")

        print(result)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        with open("transcriptions.txt", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {result}\n")

        transcription = clean_transcription(result)
        prompt = transcription
        self.loading_transcription = False
        self.loading_image = True
        if lang == "et":
            self.update_text(f"Töötlemine lõpetatud.\nSinu kirjeldus: {transcription}\nLoon pilti...")
            prompt = translate(prompt)
        else:
            self.update_text(f"Processing complete.\nYour description: {transcription}\nGenerating an image...")
        self.generate_image(prompt, lang, timestamp, transcription)


    def generate_image(self, prompt: str, lang: str, timestamp: str, transcription: str):
        image = painter(prompt=prompt + ", sharp focus, soft lighting", num_inference_steps=25 ).images[0]
        image_path = timestamp + ".png"
        image.save(image_path)
        self.loading_image = False
        self.display_image(image_path)
        self.update_listening_labels(lang, transcription)
        self.record_command()


def main():
    global transcriber, translator, tokenizer, painter, processor, est_model, lang_detector, classifier

    classifier = EncoderClassifier.from_hparams("speechbrain/lang-id-voxlingua107-ecapa")

    # transcriber = Whisper.from_pretrained("medium.en") # whispercpp
    transcriber = whisper.load_model("base.en")

    est_model_name = "TalTechNLP/espnet2_estonian"
    est_model = Speech2Text.from_pretrained(est_model_name,
                                            device="cuda",                               # for Nvidia GPUs; otherwise "cpu"
                                            lm_weight=0.6, ctc_weight=0.4, beam_size=60)

    translator_model_name = "Helsinki-NLP/opus-mt-et-en"
    tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
    translator = MarianMTModel.from_pretrained(translator_model_name)

    painter_model_name = "dreamlike-art/dreamlike-photoreal-2.0"
    painter = StableDiffusionPipeline.from_pretrained(painter_model_name, torch_dtype=torch.float16)
    painter = painter.to("cuda") # for Nvidia GPUs; otherwise use "cpu" or on Apple Silicon "mps"

    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
