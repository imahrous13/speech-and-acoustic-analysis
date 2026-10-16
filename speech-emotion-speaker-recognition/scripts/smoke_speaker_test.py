from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from predict import predict_speaker

project_root = Path(__file__).resolve().parent.parent
p = project_root / "data" / "ravdess" / "Actor_24" / "03-01-05-02-01-01-24.wav"
print('Audio exists:', p.exists())
try:
    res = predict_speaker(str(p))
    print('Result:', res)
except Exception as e:
    import traceback; traceback.print_exc()
    print('Exception:', e)
