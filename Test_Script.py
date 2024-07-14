from nnmnkwii.preprocessing import delta_features as first_der
from os.path import basename
import numpy as np
from scipy.io import wavfile
import pyworld
import pysptk
from pysptk.synthesis import MLSADF, Synthesizer
from nnmnkwii.baseline.gmm import MLPG
import IPython
from IPython.display import Audio
from joblib import load


def features_collect(source_path):
    sampling_rate, feature = wavfile.read(source_path)
    feature = feature.astype(np.float64)
    freq, timeaxis = pyworld.dio(feature, sampling_rate, frame_period=frame_quantum)
    freq = pyworld.stonemask(feature, freq, timeaxis, sampling_rate)
    spect = pyworld.cheaptrick(feature, freq, timeaxis, sampling_rate)
    mcc = pysptk.sp2mc(spect, order=n_mcc, alpha=alpha_val)
    return mcc, feature    

def test(source_path, enable_mlpg=True, vc=True):
    dim_static=30
    if enable_mlpg:
        paramgen = MLPG(gmm_model, windows=windows, diff=vc)
    else:
        paramgen = MLPG(gmm_model, windows=[(0,0, np.array([1.0]))], diff=vc)

    mcc, feature=features_collect(source_path)
    mcc0, mcc = mcc[:, 0], mcc[:, 1:]
    mcc = first_der(mcc, windows)
    mcc = paramgen.transform(mcc)
    if (not enable_mlpg) and (mcc.shape[-1] != dim_static):
        mcc = mcc[:,:dim_static]
    assert mcc.shape[-1] == dim_static
    mcc = np.hstack((mcc0[:, None], mcc))
    mcc[:, 0] = 0
    engine = Synthesizer(MLSADF(order=n_mcc, alpha=alpha_val), hopsize=hopsize)
    b = pysptk.mc2b(mcc.astype(np.float64), alpha=alpha_val)
    waveform = engine.synthesis(feature, b)

    return waveform



if __name__ == "__main__":

    sampling_rate = 48000
    alpha_val = pysptk.util.mcepalpha(sampling_rate)
    n_mcc = 30
    frame_quantum = 5
    hopsize = int(sampling_rate * (frame_quantum * 0.001))
    fft_len=pyworld.get_cheaptrick_fft_size(sampling_rate)
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]

    model_filename = input("Enter the path of the model: ")
    gmm_model = load(model_filename)

    src_path = input("Enter the path of the source audio: ")
    tgt_path = input("Enter the path of the target audio: ")
    without_MLPG = test(src_path, enable_mlpg=True)
    with_MLPG = test(src_path, enable_mlpg=False)
    _, src = wavfile.read(src_path)
    _, tgt = wavfile.read(tgt_path)
    print("Source Audio:", basename(src_path))
    IPython.display.display(Audio(src, rate=sampling_rate))
    print("Target Audio:", basename(tgt_path))
    IPython.display.display(Audio(tgt, rate=sampling_rate))
    print("With MLPG Converted Audio:")
    IPython.display.display(Audio(with_MLPG, rate=sampling_rate))
    print("Without MLPG Converted Audio:")
    IPython.display.display(Audio(without_MLPG, rate=sampling_rate))