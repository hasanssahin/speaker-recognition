import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import time
import librosa
import librosa.display

def calculate_histogram(audio_file):

    sample_rate=48000
    # Upload audio file
    audio, _ = sf.read(audio_file)

    # Calculate range of sampling points
    sample_period = 1.0 / sample_rate
    total_samples = len(audio)
    duration = total_samples / sample_rate

    # Calculate frequency spectrum
    frequencies = np.fft.fftfreq(total_samples, sample_period)
    magnitudes = np.abs(np.fft.fft(audio))

    # Create the histogram
    plt.hist(frequencies, bins=50, weights=magnitudes)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.title('Audio Histogram')

    # Save the histogram
    timestamp = str(int(time.time()))  # Create a unique timestamp
    filename = 'histogram_' + timestamp + '.png'  # Generate filename
    save_path = os.path.join('../histograms', filename)
    plt.savefig(save_path)

    return save_path



def calculate_mel_spectrogram(audio_file_path):
    sr=48000
    
    # Upload the audio file
    signal, sr = librosa.load(audio_file_path, sr=sr)

    # Calculate the signal spectrogram
    spec = np.abs(librosa.stft(signal))

    # Calculate mel properties
    mel_spec = librosa.feature.melspectrogram(S=spec, sr=sr)

    # Get the logarithm of powers at Mel frequencies
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Draw the histogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    # Save the histogram
    timestamp = str(int(time.time()))  # Create a unique timestamp
    filename = 'spectrogram_' + timestamp + '.png'  # Generate filename
    save_path = os.path.join('../spectrograms', filename)
    plt.savefig(save_path)

    return save_path


def plot_audio_signal(audio_file_path):
    
    sr=48000
    # Upload the audio file
    y, sr = librosa.load(audio_file_path, sr=sr)

    # Draw the audio signal
    plt.figure(figsize=(15, 5))
    plt.plot(y)
    plt.title("Audio Signal")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")

    # Save the histogram
    timestamp = str(int(time.time()))  # Create a unique timestamp
    filename = 'signal_' + timestamp + '.png'  # Generate filename
    save_path = os.path.join('../signals', filename)
    plt.savefig(save_path)

    return save_path
