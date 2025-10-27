import torchaudio

# load audio file
waveform, sample_rate = torchaudio.load('test.wav')

# resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
# waveform = resampler(waveform)

# Preprocessing - Mel-frequency cepstral coefficients (MFCCs)
mfcc_transform =torchaudio.transforms.MFCC(sample_rate=sample_rate)
mfcc = mfcc_transform(waveform)

