import tensorflow as tf
import logging
import numpy as np
from datetime import datetime
import scipy.io as sio
import os
import time
import argparse
import AINN as Model
import DataLoader

tf.config.set_visible_devices([], 'GPU')

logging.getLogger('tensorflow').setLevel(logging.ERROR)
DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--room_number", type=int, default=1, help="specify the room number")
parser.add_argument("--speaker_set", type=int, default=0, help="specify the speaker set")
parser.add_argument("--input_file", type=str, default="", help="specify the data directory")
# parser.add_argument("--file_name_prefix", type=str, default="", help="specify the file name prefix")
# parser.add_argument("--microphone_number", type=int, default=30, help="specify the speaker set")
parser.add_argument("--output_file", type=str, default="", help="specify the results directory")
args = parser.parse_args()

#####################################################################################################################
# Load the RTF data
#####################################################################################################################
# Specify the room and zone in the UTS RIR database

# room_number = args.room_number
# data_dir = args.data_dir
# file_name_prefix = args.file_name_prefix
# results_dir = args.results_dir
# room_number = args.room_number
# if room_number == 1:
#     data_dir = '../UTS RIR Database/Anechoic Room/ZoneE/CircularMicrophoneArray/'
#     file_name_prefix = 'AnechoicRoom_ZoneE_CircularMicrophoneArray'
#     print("Anechoic Room\n\n")
#     results_dir = '../Results/AnechoicRoom/'
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
# elif room_number == 2:
#     data_dir = '../UTS RIR Database/Hemi-anechoic Room/ZoneE/CircularMicrophoneArray/'
#     file_name_prefix = 'HemiAnechoicRoom_ZoneE_CircularMicrophoneArray'
#     print("\n\nHemi-Anechoic Room\n\n")
#     results_dir = '../Results/HemiAnechoicRoom/'
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
# elif room_number == 3:
#     data_dir = '../UTS RIR Database/Small Meeting Room/ZoneE/CircularMicrophoneArray/'
#     file_name_prefix = 'SmallMeetingRoom_ZoneE_CircularMicrophoneArray'
#     print("Small Meeting Room\n\n")
#     results_dir = '../Results/SmallMeetingRoom/'
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
# elif room_number == 4:
#     data_dir = '../UTS RIR Database/Medium Meeting Room/ZoneE/CircularMicrophoneArray/'
#     file_name_prefix = 'MediumMeetingRoom_ZoneE_CircularMicrophoneArray'
#     print("Medium Meeting Room\n\n")
#     results_dir = '../Results/MediumMeetingRoom/'
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
# elif room_number == 5:
#     data_dir = '../UTS RIR Database/Large Meeting Room/ZoneE/CircularMicrophoneArray/'
#     file_name_prefix = 'LargeMeetingRoom_ZoneE_CircularMicrophoneArray'
#     print("Large Meeting Room\n\n")
#     results_dir = '../Results/LargeMeetingRoom/'
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
# elif room_number == 6:
#     data_dir = '../UTS RIR Database/Shoe-box Room/ZoneE/CircularMicrophoneArray/'
#     file_name_prefix = 'ShoeBoxRoom_ZoneE_CircularMicrophoneArray'
#     print("Shoe-box Room\n\n")
#     results_dir = '../Results/ShoeBoxRoom/'
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
# elif room_number == 7:
#     data_dir = '../UTS RIR Database/Open-plan Office/ZoneE/CircularMicrophoneArray/'
#     file_name_prefix = 'OpenSpace_ZoneE_CircularMicrophoneArray'
#     print("Open-plan Office\n\n")
#     results_dir = '../Results/OpenSpace/'
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
# else:
#     raise ValueError("Room number not found!")

# Specify the loudspeaker index in the UTS RIR database
# speaker_index = [7]  # loudspeaker position
# speaker_index = range(1,61)
speaker_set = args.speaker_set
speaker_index = [speaker_set]
# if speaker_set == 1:
#     speaker_index = range(1, 6)
# elif speaker_set == 2:
#     speaker_index = range(6, 11)
# elif speaker_set == 3:
#     speaker_index = range(11, 16)
# elif speaker_set == 4:
#     speaker_index = range(16, 21)
# elif speaker_set == 5:
#     speaker_index = range(21, 26)
# elif speaker_set == 6:
#     speaker_index = range(26, 31)
# elif speaker_set == 7:
#     speaker_index = range(31, 36)
# elif speaker_set == 8:
#     speaker_index = range(36, 41)
# elif speaker_set == 9:
#     speaker_index = range(41, 46)
# elif speaker_set == 10:
#     speaker_index = range(46, 51)
# elif speaker_set == 11:
#     speaker_index = range(51, 56)
# elif speaker_set == 12:
#     speaker_index = range(56, 61)
# else:
#     raise ValueError("Speaker set not specified!")

# Load the RTF data
# RTF = myDataLoader.loadRTF(data_dir, file_name_prefix, speaker_index)
# # save the RTF data
# RTF_path = './RTF/RTF_' + file_name_prefix + '_Speaker_set_' + str(speaker_set) + '.mat'
# sio.savemat(RTF_path, RTF)

RTF_file = args.input_file
RTF = sio.loadmat(RTF_file)

Microphone_number = 30
Microphone_index = np.arange(0, 30, 30/Microphone_number)
Microphone_index = Microphone_index.astype(int)
# Define the coordinates of the microphones and the gradients
# coordinates of the microphones where the sound pressure is measured
Microphone_coordinates = RTF['Microphone_coordinates'][Microphone_index, :]
# coordinates of the points where the sound pressure is estimated
p_pred_coordinates = RTF['Sound_pressure_prediction_coordinates']
# coordinates of the points where the pressure gradient is estimated
grad_pred_coordinates = RTF['Gradients_prediction_coordinates']
# coordinates of the points where the PDE is solved
PDE_coordinates = RTF['PDE_coordinates']
# convert to tensors
mic_xy = tf.convert_to_tensor(Microphone_coordinates, dtype=tf.float64)
p_pred_xy = tf.convert_to_tensor(p_pred_coordinates, dtype=tf.float64)
grad_pred_xy = tf.convert_to_tensor(grad_pred_coordinates, dtype=tf.float64)
pde_xy = tf.convert_to_tensor(PDE_coordinates, dtype=tf.float64)

#####################################################################################################################
# define the network and training parameters
layers = 2  # layer 1
num_epoch = 100 * 1000  # training epoches
#####################################################################################################################
# iterate over different frequencies
# Specify the frequency range and interval to be used
freq = np.arange(100, 4001, 20)
# freq = np.array([1084, 1728, 2316, 2490, 2878, 3164, 3422, 3796, 3902, 3956])
sound_speed = 340  # speed of sound
for spk in speaker_index:
    AINNResults = {
        'microphone_number': Microphone_number,
        'microphone_index': Microphone_index,
        'p_coordinates_mic': RTF['Microphone_coordinates'],
        'p_coordinates_pde': PDE_coordinates,
        'grad_coordinates_mic': RTF['Microphone_coordinates'],
        'grad_coordinates': grad_pred_coordinates,
        'frequencies': freq,
        'sampling_rate': 48000,
        'speaker_set': speaker_set,

        'p_true_values': np.zeros((RTF['Microphone_coordinates'].shape[0], freq.shape[0]), dtype=complex),
        'p_predicted_mic': np.zeros((RTF['Microphone_coordinates'].shape[0], freq.shape[0]), dtype=complex),
        'p_predicted_pde': np.zeros((PDE_coordinates.shape[0], freq.shape[0]), dtype=complex),

        'grad_predicted_x_mic': np.zeros((RTF['Microphone_coordinates'].shape[0], freq.shape[0]), dtype=complex),
        'grad_predicted_y_mic': np.zeros((RTF['Microphone_coordinates'].shape[0], freq.shape[0]), dtype=complex),

        'grad_true_values': np.zeros((grad_pred_coordinates.shape[0], freq.shape[0]), dtype=complex),
        'grad_predicted_x': np.zeros((grad_pred_coordinates.shape[0], freq.shape[0]), dtype=complex),
        'grad_predicted_y': np.zeros((grad_pred_coordinates.shape[0], freq.shape[0]), dtype=complex),
        # 'Best_model_real': [None] * freq.shape[0],
        # 'Best_model_imag': [None] * freq.shape[0],
    }
    index = 0
    for f in freq:
        f_index = np.where(f == RTF['frequencies'][0])[0][0]
        wave_number = 2 * np.pi * f / sound_speed
        nodes = int(np.ceil(wave_number * 0.12))
        # nodes = 6
        ##############################################################################################################
        # prepare the training data
        ##############################################################################################################
        # read the true values at the measurement microphones to train the model
        mic_true_values = np.zeros((Microphone_number, 1), dtype=complex)  # important to initialize the ndarray as a 2D array
        mic_true_values[0:Microphone_number,0] = RTF['sound_pressure'][Microphone_index, f_index]
        mic_true_values_real = mic_true_values.real
        mic_true_max_real = np.max(np.abs(mic_true_values_real))
        mic_true_values_real = mic_true_values_real/mic_true_max_real # normalize the true values for consistent training
        mic_tue_values_imag = mic_true_values.imag
        mic_true_max_imag = np.max(np.abs(mic_tue_values_imag))
        mic_tue_values_imag = mic_tue_values_imag/mic_true_max_imag
        ##############################################################################################################
        # convert to tensor
        mic_true_values_real = tf.convert_to_tensor(mic_true_values_real, dtype=tf.float64)
        mic_tue_values_imag = tf.convert_to_tensor(mic_tue_values_imag, dtype=tf.float64)
        wave_number = tf.convert_to_tensor(wave_number, dtype=tf.float64)
        ##############################################################################################################
        # initialize the variables for saving the training results
        min_loss_real = float('inf')
        min_loss_imag = float('inf')
        best_model_real = None
        best_model_imag = None
        ##############################################################################################################
        for version in range(1, 11):
            # clear tensorflow sessions
            tf.keras.backend.clear_session()
            #################################################################################################
            # training and predicting the real part of the sound field
            #################################################################################################
            print('Training the real part...')
            optim_real = tf.keras.optimizers.Adam(learning_rate=1e-3)  # ADAM optimizer
            model_real = Model.AINNModel2D(layers=layers, neurons=nodes)
            loss_data_real, loss_pde_real = model_real.train(optim=optim_real, mic_coordinates=mic_xy,
                                                             mic_true_values=mic_true_values_real,
                                                             pde_coordinates=pde_xy,
                                                             wave_number=wave_number, num_epoch=num_epoch)

            now = datetime.now()  # current date and time
            now = now.strftime("%H:%M:%S")
            # print('\nResults for real part:')
            print('Input file:', RTF_file, '\nSpeaker:', spk, 'Frequency:', f, 'Hz', 'Version', version, 'Time', now,
                  'Layers', layers, 'Nodes', nodes, 'Data loss:', loss_data_real[-1],'PDE loss', loss_pde_real[-1])
            # print(loss_data_real)
            # print(loss_pde_real)

            if loss_data_real[-1] < min_loss_real:
                min_loss_real = loss_data_real[-1]
                best_model_real = model_real

            #################################################################################################
            # training and predicting the imaginary part of the sound field
            #################################################################################################
            print('Training the imaginary part...')
            optim_imag = tf.keras.optimizers.Adam(learning_rate=1e-3)  # ADAM optimizer
            model_imag = Model.AINNModel2D(layers=layers, neurons=nodes)
            loss_data_imag, loss_pde_imag = model_imag.train(optim=optim_imag, mic_coordinates=mic_xy,
                                                             mic_true_values=mic_tue_values_imag,
                                                             pde_coordinates=pde_xy,
                                                             wave_number=wave_number, num_epoch=num_epoch)

            now = datetime.now()  # current date and time
            now = now.strftime("%H:%M:%S")
            # print('\nResults for imaginary part:')
            print('Input file:', RTF_file, '\nSpeaker:', spk, 'Frequency:', f, 'Hz', 'Version', version, 'Time', now,
                  'Layers', layers, 'Nodes', nodes, 'Data loss:', loss_data_imag[-1], 'PDE loss', loss_pde_imag[-1])
            # print(loss_data_imag)
            # print(loss_pde_imag)

            if loss_data_imag[-1] < min_loss_imag:
                min_loss_imag = loss_data_imag[-1]
                best_model_imag = model_imag

        print('Results for the best model for Input file:', RTF_file, '\nSpeaker:', spk, 'Frequency:', f, 'Hz', 'Time', now,
                  'Layers', layers, 'Nodes', nodes)
        print('Real part:')
        print(loss_data_real)
        print(loss_pde_real)
        print('Imaginary part:')
        print(loss_data_imag)
        print(loss_pde_imag)
        #################################################################################################
        # Predicting the real part of the sound field using the best model
        #################################################################################################
        # print('\nPredicting the sound pressure...')
        # predict the real part of sound pressure
        p_pred_real, _, _ = best_model_real.predict(RTF['Microphone_coordinates'])
        p_pred_imag, _, _ = best_model_imag.predict(RTF['Microphone_coordinates'])
        p_pred_values = p_pred_real.numpy() * mic_true_max_real + 1j * p_pred_imag.numpy() * mic_true_max_imag
        AINNResults['p_predicted_mic'][:,index] = p_pred_values.flatten()

        p_pred_pde_real, _, _ = best_model_real.predict(pde_xy)
        p_pred_pde_imag, _, _ = best_model_imag.predict(pde_xy)
        p_pred_pde_values = p_pred_pde_real.numpy() * mic_true_max_real + 1j * p_pred_pde_imag.numpy() * mic_true_max_imag
        AINNResults['p_predicted_pde'][:, index] = p_pred_pde_values.flatten()

        _, dp_dx_real_0, dp_dy_real_0 = best_model_real.predict(RTF['Microphone_coordinates'])
        _, dp_dx_imag_0, dp_dy_imag_0 = best_model_imag.predict(RTF['Microphone_coordinates'])
        dp_dx_values_0 = dp_dx_real_0.numpy() * mic_true_max_real + 1j * dp_dy_real_0.numpy() * mic_true_max_imag
        dp_dy_values_0 = dp_dx_imag_0.numpy() * mic_true_max_real + 1j * dp_dy_imag_0.numpy() * mic_true_max_imag
        AINNResults['grad_predicted_x_mic'][:, index] = dp_dx_values_0.flatten()
        AINNResults['grad_predicted_y_mic'][:, index] = dp_dy_values_0.flatten()

        #################################################################################################
        # Predicting the imaginary part of the sound field using the best model
        #################################################################################################
        # print('\nPredicting the sound pressure gradient...')
        _, dp_dx_real, dp_dy_real = best_model_real.predict(grad_pred_xy)
        _, dp_dx_imag, dp_dy_imag = best_model_imag.predict(grad_pred_xy)
        dp_dx_values = dp_dx_real.numpy() * mic_true_max_real + 1j * dp_dx_imag.numpy()* mic_true_max_imag
        dp_dy_values = dp_dy_real.numpy() * mic_true_max_real + 1j * dp_dy_imag.numpy()* mic_true_max_imag
        AINNResults['grad_predicted_x'][:,index] = dp_dx_values.flatten()
        AINNResults['grad_predicted_y'][:,index] = dp_dy_values.flatten()

        # retrieve the true values
        AINNResults['p_true_values'][:,index] = RTF['sound_pressure'][:, f_index]
        AINNResults['grad_true_values'][:,index] = RTF['pressure_gradients'][:, f_index]
        # save the best models
        # PINNResults['Best_model_real'][index] = best_model_real
        # PINNResults['Best_model_imag'][index] = best_model_imag
        index = index + 1
        print('Frequency', f, 'Hz completed!\n')
        ##############################################################################################################

    # output_file = 'PINNResults_Room_' + str(room_number) + '_Speaker_' + str(spk) + '.mat'
    output_file = args.output_file
    sio.savemat(output_file, AINNResults)




