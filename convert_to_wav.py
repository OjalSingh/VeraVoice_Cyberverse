# convert_to_wav.py
import os
from pydub import AudioSegment

def convert_folder(src_folder, dst_folder, target_sr=16000):
    os.makedirs(dst_folder, exist_ok=True)
    for fn in os.listdir(src_folder):
        src = os.path.join(src_folder, fn)
        if not os.path.isfile(src): continue
        name, _ = os.path.splitext(fn)
        dst = os.path.join(dst_folder, f"{name}.wav")
        try:
            audio = AudioSegment.from_file(src)
            audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)  # 16-bit
            audio.export(dst, format="wav")
            print("Saved:", dst)
        except Exception as e:
            print("Error", src, e)

if __name__ == "__main__":
    convert_folder("data/real", "data/real_wav")
    convert_folder("data/fake", "data/fake_wav")
