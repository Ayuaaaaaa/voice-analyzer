"""Print gap signal statistics for every audio file in a folder."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from pipeline import noise, speech  # noqa: E402
from pipeline.audio import load_audio  # noqa: E402

for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.ogg"))):
    audio, _ = load_audio(path)
    sp = speech.analyze_speech(audio)
    ns = speech.non_speech_mask(len(audio), sp.speech_regions)
    print(os.path.basename(path), noise.gap_signal_stats(audio, ns))
