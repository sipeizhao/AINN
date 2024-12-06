import tensorflow as tf
import numpy as np


class AINNModel2D(tf.keras.Model):
    def __init__(self, layers=4, neurons=1):
        super(AINNModel2D, self).__init__()
        # Initialize the layers in the model
        self.model_layers = [tf.keras.layers.Dense(neurons, activation='tanh', kernel_initializer='glorot_normal') for _ in range(layers)]
        self.output_layer = tf.keras.layers.Dense(1)  # Output layer

    def __call__(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return self.output_layer(x)

    ###########################################################################################################
    # calculate the data loss
    def calculate_data_loss(self, mic_coordinates, mic_true_values):
        data_pred = self(mic_coordinates)
        loss_data = tf.reduce_mean(tf.square(data_pred - mic_true_values))
        return loss_data

    ###########################################################################################################
    # calculate the PDE loss
    def calculate_pde_loss(self, pde_coordinates, wave_number):
        x, y = pde_coordinates[:, 0:1], pde_coordinates[:, 1:2]
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            tape1.watch(y)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)
                tape2.watch(y)
                pde_pred = self(tf.stack([x[:, 0], y[:, 0]], axis=1))
            dp_dx = tape2.gradient(pde_pred, x)
            dp_dy = tape2.gradient(pde_pred, y)
        d2p_dx2 = tape1.gradient(dp_dx, x)
        d2p_dy2 = tape1.gradient(dp_dy, y)
        del tape1, tape2
        pde_eqn = (d2p_dx2 + d2p_dy2) / (wave_number**2) + pde_pred  # the Helmholtz equation
        loss_pde = tf.reduce_mean(tf.square(pde_eqn))
        return loss_pde

    ###########################################################################################################
    # calculate the gradient with respect to the data loss and the PDE loss
    def calculate_gradients(self, mic_coordinates, mic_true_values, pde_coordinates, wave_number):
        with tf.GradientTape(persistent=True) as tape:
            loss_data = self.calculate_data_loss(mic_coordinates, mic_true_values)  # data loss
            loss_pde = self.calculate_pde_loss(pde_coordinates, wave_number)  # pde loss
            loss_total = loss_data + loss_pde  # total loss
        grads = tape.gradient(loss_total, self.trainable_variables)  # the gradient
        del tape
        return loss_data, loss_pde, grads

    @tf.function
    def model_fit(self, optim, mic_coordinates, mic_true_values, pde_coordinates, wave_number):
        loss_data, loss_pde, grad = self.calculate_gradients(mic_coordinates, mic_true_values, pde_coordinates, wave_number)
        optim.apply_gradients(zip(grad, self.trainable_variables))
        return loss_data, loss_pde

    def train(self, optim, mic_coordinates, mic_true_values, pde_coordinates, wave_number, num_epoch):
        loss_data_history = []  # data loss learning curve
        loss_pde_history = []   # PDE loss learning curve
        prev_loss = float('inf')  # Initialize with a high value
        same_loss_count = 0  # Count of consecutive same losses
        true_db = 10 * np.log10(tf.reduce_mean(tf.square(mic_true_values))) // 0.1 / 10  # true values in dB
        print('true_db', true_db)
        #############################################################################################################
        for ii in range(num_epoch):
            loss_data, loss_pde = self.model_fit(optim, mic_coordinates, mic_true_values, pde_coordinates, wave_number)
            if ii % 1000 == 0:
                loss_data_db = (10 * np.log10(loss_data) - true_db) // 0.1 / 10
                loss_pde_db = (10 * np.log10(loss_pde) - true_db) // 0.1 / 10
                loss_data_history.append(loss_data_db)
                loss_pde_history.append(loss_pde_db)

                # Check if the current loss is the same as the previous loss
                if loss_data_db == prev_loss:
                    same_loss_count += 1
                else:
                    same_loss_count = 0

                # Update the previous loss
                prev_loss = loss_data_db

                # If the loss is the same for 5 consecutive iterations, stop the loop
                if same_loss_count == 5:
                    print("Data Loss hasn't changed for the last 5 iterations. Stopping training. The algorithm converged after {} epoches.".format(ii))
                    break

        return loss_data_history, loss_pde_history

    ##############################################################################################################
    # Predict the sound pressure and its gradient at the given coordinates
    def predict(self, test_coordinates):
        # convert the input test_coordinates to a tf.Tensor if it is not already a tf.Tensor
        if not isinstance(test_coordinates, tf.Tensor):
            test_coordinates = tf.convert_to_tensor(test_coordinates, dtype=tf.float64)

        x, y = test_coordinates[:, 0:1], test_coordinates[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            pinn_pred = self(tf.stack([x[:, 0], y[:, 0]], axis=1))
        dp_dx = tape.gradient(pinn_pred, x)
        dp_dy = tape.gradient(pinn_pred, y)
        del tape
        return pinn_pred, dp_dx, dp_dy

