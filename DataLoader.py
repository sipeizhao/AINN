import numpy as np
import os
import soundfile as sf

# calculate the single-sided frequency spectrum of a signal x
def FFT(x, fs, nfft):
    # Perform FFT
    X = np.fft.fft(x, n=nfft)
    # Keep the first half of the FFT results and normalize
    X = X[0:int(nfft/2) + 1]/nfft
    # Multiply by 2 to take into account the other half of the spectrum
    X[1:int(nfft/2)] = 2 * X[1:int(nfft/2)]

    # Calculate the frequencies
    freq = fs*np.array(range(0, int(nfft/2) + 1))/nfft
    return X, freq

def loadRTF(data_dir, file_name_prefix, speaker_index):
    RTF = {}
    # Specify the coordinates of the microphones
    outer_mic_radius = 0.12
    inner_mic_radius = 0.10
    RTF['Microphone_coordinates'] = np.zeros((60, 2))
    RTF['Microphone_coordinates'][0:30, 0] = -outer_mic_radius * np.sin(np.arange(0, 30) * 2 * np.pi / 30)
    RTF['Microphone_coordinates'][0:30, 1] = outer_mic_radius * np.cos(np.arange(0, 30) * 2 * np.pi / 30)
    RTF['Microphone_coordinates'][30:60, 0] = -inner_mic_radius * np.sin(np.arange(0, 30) * 2 * np.pi / 30)
    RTF['Microphone_coordinates'][30:60, 1] = inner_mic_radius * np.cos(np.arange(0, 30) * 2 * np.pi / 30)
    RTF['Sound_pressure_prediction_coordinates'] = RTF['Microphone_coordinates'][30:60, :]
    # Specify the coordinates of the gradients
    RTF['Gradients_prediction_coordinates'] = np.zeros((30, 2))
    RTF['Gradients_prediction_coordinates'][:, 0] = -(outer_mic_radius + inner_mic_radius) / 2 * np.sin(np.arange(0, 30) * 2 * np.pi / 30)
    RTF['Gradients_prediction_coordinates'][:, 1] = (outer_mic_radius + inner_mic_radius) / 2 * np.cos(np.arange(0, 30) * 2 * np.pi / 30)

    # Specify the coordinates of the PDE points
    pde_x, pde_y = np.meshgrid(np.arange(-0.12, 0.13, 0.01), np.arange(-0.12, 0.13, 0.01))
    pde_x = pde_x.flatten()
    pde_y = pde_y.flatten()
    index = np.where(np.sqrt(pde_x ** 2 + pde_y ** 2) <= 0.13)
    RTF['PDE_coordinates'] = np.zeros((len(index[0]), 2))
    RTF['PDE_coordinates'][:, 0] = pde_x[index]
    RTF['PDE_coordinates'][:, 1] = pde_y[index]
    RTF['fs'] = 48000
    # only use the first 0.5 seconds of the RIRs, which is long enough for the RTF calculation
    RTF['nfft'] = int(RTF['fs'] * 0.5)

    for spk_index in speaker_index:
        RTF['Speaker_' + str(spk_index)] = {
            'sound_pressure': np.zeros((60, int(RTF['nfft'] / 2) + 1), dtype=complex),
            'pressure_gradients': np.zeros((30, int(RTF['nfft'] / 2) + 1), dtype=complex),
        }
        for mic_index in range(0, 60):
            file_name = file_name_prefix + '_L' + str(spk_index) + '_M' + str(mic_index + 1) + '.wav'
            audio_waveform, fs = sf.read(data_dir + file_name)
            x = audio_waveform[0:int(0.5 * fs)]  # only use the first 0.5 seconds of the audio file
            nfft = len(x)
            X, freq = FFT(x, fs, nfft)
            RTF['Speaker_' + str(spk_index)]['sound_pressure'][mic_index, :] = X
            # RTF['Speaker_' + str(spk_index)]['sound_pressure'].append(X)
        grads = RTF['Speaker_' + str(spk_index)]['sound_pressure'][0:30, :] - RTF['Speaker_' + str(spk_index)][
                                                                                  'sound_pressure'][30:60, :]
        grads = grads / (outer_mic_radius - inner_mic_radius)
        RTF['Speaker_' + str(spk_index)]['pressure_gradients'] = grads
    RTF['frequencies'] = freq
    return RTF