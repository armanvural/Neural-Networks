import h5py
import numpy as np
import numpy.ma as ma
import sys
import matplotlib.pyplot as plt
import random
import math





question = input("Enter Relevant Question Number to Display its Respective Output \n")


def arman_budunoglu_21602635_hw3(question):

    if question == "1":

        print("Question 1: ")

        # Retrieving the training Data from the h5 file as numpy arrays
        hf = h5py.File('assign3_data1.h5', 'r')
        ls = list(hf.keys())

        print('List of datasets in this file ', ls)
        # Obtaining the data from the h5 file
        data = np.array(hf.get('data'))
        invXForm = np.array(hf.get('invXForm'))
        xForm = np.array(hf.get('xForm'))
        # Checking the shape of the contents
        # print(data.shape) # (10240, 3, 16, 16)
        # print(invXForm.shape) # (105, 768)
        # print(xForm.shape) # (768, 105)
        # Data's 2nd dimension contains RGB values

        print('Question1:Part:A')
        # Preprocessing the data
        # Converting data to grayscale using the luminosity model
        Y = (0.2126*data[:, 0, :, :]) + (0.7152 * data[:, 1, :, :]) + (0.0722 * data[:, 2, :, :])
        #Y.shape = (10240,16,16)
        # Normalizing the data
        # Obtaining the mean of each image
        mean_set = Y.reshape(10240,256)
        mean_array = np.mean(mean_set,axis=1)
        # Subtracting the mean of each image from itself
        for i in range(int(len(mean_array))):
            mean_set[i][:] = mean_set[i][:] - mean_array[i]

        mean_set = mean_set.reshape(10240,16,16)
        # Obtaining the std of the set to clip +- 3 stds
        std_set = np.std(mean_set)
        min_clip = std_set*-3
        max_clip = std_set*3
        # Clipping the data between +-3 std values
        clippers = np.clip(mean_set,min_clip,max_clip)
        # Adjusting the clipped data by min-max normalization with interval [0.1 0.9]
        # In order to ensure all features have the same scale I chose min-max norm.
        min_scale = 0.1
        max_scale = 0.9
        clipped_max = np.max(clippers)
        clipped_min = np.min(clippers)
        # Normalization for 0,1 range
        adjust_scale = (clippers - clipped_min)/(clipped_max-clipped_min)
        # Normalizing to the given range
        normal_data = (adjust_scale * (max_scale-min_scale)) + min_scale

        #  plots Q1/A
        # Obtaining random samples index
        random_patch = np.random.permutation(10240)
        # Printing RGB Images
        usher_RGB = 0
        plt.figure()
        for i in range(200):
            usher_RGB += 1
            RGB_ims = data[random_patch[i]]
            # # Reshaping the data to fit imshow function's format
            rgb_data = np.transpose(RGB_ims, (2, 1, 0))
            plt.subplot(10, 20, usher_RGB)
            plt.imshow(rgb_data)
            plt.axis('off')
        plt.savefig('Q1_A_RGB.png', bbox_inches='tight')
        plt.show()

        # Printing Normalized Images
        usher_norm = 0
        plt.figure()
        for k in range(200):
            usher_norm += 1
            norm_ims = normal_data[random_patch[k]]
            plt.subplot(10, 20, usher_norm)
            plt.imshow(norm_ims.T, cmap='gray')
            plt.axis('off')

        plt.savefig('Q1_A_norm.png', bbox_inches='tight')
        plt.show()

        print('Question1:Part:B')
        # Network Initializations
        # Normalized data is reshaped to (10240,256)
        data_in = normal_data.reshape(10240,256)
        # Network parameters
        lamb_param = 0.0005
        rho = 0.05
        beta = 0.01
        # Data is of size L_in * N where L_in = 16*16 = 256 and N = 10240
        L_in = data_in.shape[1]
        L_hid = 64
        # Number of input neurons
        L_pre_1 = data.shape[0] # N=10240
        sample = L_pre_1
        # Number of Neurons on the input to hidden layer
        L_post_1 = L_hid
        # Initializing weight and biases of the first layer
        w_0 = Q1_weight_bias_interv(L_pre_1,L_post_1)
        weights_1 = np.random.uniform(-w_0, w_0, (256,L_hid))
        bias_1 = np.random.uniform(-w_0, w_0, (1,L_hid))
        # Initializing weight and biases of the hidden to output layer
        L_pre_2 = L_post_1
        L_post_2 = L_pre_1
        # The same w_0 value is obtained as the previous layer
        weights_2 = np.random.uniform(-w_0, w_0, (L_hid,256))
        bias_2 = np.random.uniform(-w_0, w_0, (1,256))

        # Creating the W_e vector that contains all weights and biases
        W_e = weights_1, weights_2, bias_1, bias_2
        # Creating a params structure that contains parameters of the network
        params = L_in, L_hid, lamb_param, beta, rho
        # In order to obtain the cost function we first need to obtain rho_hat_j which represents a vector that
        # contains average activation of hidden layer neuron j for all hidden layer neurons
        # Assuming sigmoid activation since we expect data to be btw 0,1
        # After utilizing the necessary steps and tidying them in to a single function I implemented the following
        # aeCost function is implemented and it returns overall cost along with network weight and bias gradients
        # Unsupervised Learning
        lrn = 1
        epochs = 1000
        error_metric = list()
        for epoch in range(epochs):
            W_en = weights_1, weights_2, bias_1, bias_2
            cost, dW1, dB1, dW2, dB2 = aeCost(W_en, data_in, params)
            # Updating Weights and Biases
            weights_2 -= dW2 * lrn
            bias_2 -= dB2 * lrn
            weights_1 -= dW1 * lrn
            bias_1 -= dB1 * lrn
            # Cost over Epochs
            error_metric.append(cost)
            if int(epoch) % 100 == 0:
                print('#Epoch:' + str(epoch) + '--> Cost = ' + str(cost))

        # Trained Weights and Biases. Trained W_E vector contains the trained network weights and Biases
        W_enc_trained = weights_1, weights_2, bias_1, bias_2

        # Plotting Cost vs Epochs
        plt.figure()
        plt.plot(error_metric)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.title('Epoch vs Cost')
        plt.legend(['Beta =' + str(beta) + ' Rho=' + str(rho)])
        plt.savefig('Q1_B_cost_metric.png')
        plt.show()

        # Examining the performance of the network by choosing random sample of images
        # and forwarding it to the network. To assess how well network extracts feautures
        lucky_charm = 12
        data_plot = data_in.reshape(10240, 16, 16)
        # Getting the OG Image Sample Batch
        og_batch = data_plot[lucky_charm:lucky_charm + 4][:][:]
        # Forwarding the samples to the trained network
        network_image = data_in[lucky_charm:lucky_charm + 4][:]
        network_pred = feature_ext(network_image,W_enc_trained,params)
        # Obtaining the extracted features form the network
        extract_plot = network_pred.reshape(4,16,16)

        # Plotting original images along with extracted feautures from the network
        #  Original Normalized Samples plot
        plt.figure()
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(og_batch[i].T, cmap='gray')
            plt.axis('off')

        plt.title('Original Images')
        plt.savefig('Q1_B_OG_ims.png')
        plt.show()

        #  Network Output
        #  Extracted Features
        plt.figure()
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(extract_plot[i].T, cmap='gray')
            plt.axis('off')

        plt.title('Extracted Features')
        plt.savefig('Q1_B_extract_ims.png')
        plt.show()

        print('Q1:Part:C')
        # Reshaping the hidden weights to fit imshow format
        w_t = weights_1.T
        weights_hid = w_t.reshape(64, 16, 16)
        # Plotting the hidden weight visualizations
        plt.figure()
        for k in range(L_hid):
            plt.subplot(8, 8, k + 1)
            plt.imshow(weights_hid[k].T, cmap='gray')
            plt.axis('off')
        plt.savefig('Q1_C_hidden_vis.png', bbox_inches='tight')
        plt.show()

        print('Q1:Part:D')
        # Created a network_train function to implement the learning algorithm along with all the steps in B for
        # for different Lhid and rho paramaters.

        # Affect of Lhid on the hidden features observed on 3 different Lhid parameters
        # Created a network_train function to implement the learning algorithm along with all the steps in B for
        # for different Lhid and lambda paramaters.
        # rho and Beta parameters are fixed
        rho = 0.05
        beta = 0.01
        # Keeping lambda fixed for L_hid networks to better assess the affect of Lhid
        hid_net_lambda = 0.0005
        # Lhid networks
        # L_hid Low
        L_hid_low = 10
        low_neuron_train = network_solver(data_in, L_hid_low, hid_net_lambda, rho, beta)

        # Formatting the low_hidden weights to plot the visualization
        low_hidden = hidden_layer_features(10, low_neuron_train)
        # Plotting Low_Neuron Hidden Weight Visualizations
        # Plotting the hidden weight visualizations
        plt.figure()
        for k in range(10):
            plt.subplot(2, 5, k + 1)
            plt.imshow(low_hidden[k].T, cmap='gray')
            plt.axis('off')
        plt.savefig('Q1_D1_L.png', bbox_inches='tight')
        plt.show()

        # L_hid_medium
        L_hid_medium = 50
        medium_neuron_train = network_solver(data_in, L_hid_medium, hid_net_lambda, rho, beta)
        # Formatting the med_hidden weights to plot the visualization
        med_hidden = hidden_layer_features(50, medium_neuron_train)

        # Plotting Med_Neuron Hidden Weight Visualizations
        # Plotting the hidden weight visualizations
        plt.figure()
        for k in range(50):
            plt.subplot(5, 10, k + 1)
            plt.imshow(med_hidden[k].T, cmap='gray')
            plt.axis('off')
        plt.savefig('Q1_D1_M.png', bbox_inches='tight')
        plt.show()

        # L_hid_high
        L_hid_high = 90
        high_neuron_train = network_solver(data_in, L_hid_high, hid_net_lambda, rho, beta)
        # Formatting the high_hidden weights to plot the visualization
        high_hidden = hidden_layer_features(90, high_neuron_train)
        # Plotting High_Neuron Hidden Weight Visualizations
        # Plotting the hidden weight visualizations
        plt.figure()
        for k in range(90):
            plt.subplot(9, 10, k + 1)
            plt.imshow(high_hidden[k].T, cmap='gray')
            plt.axis('off')
        plt.savefig('Q1_D1_H.png', bbox_inches='tight')
        plt.show()

        # Lambda networks
        # Repeating the Same Procedure for lambda parameter
        # Keeping the L_hid parameter fixed to better assess lambda parameters overall effect
        h_neuron_num = 50
        # Lambda Low
        low_lambda = 0
        low_lambda_train = network_solver(data_in, h_neuron_num, low_lambda, rho, beta)
        low_lambda_hidden = hidden_layer_features(50, low_lambda_train)

        # Plotting Lambda Low Hidden Weight Visualizations
        # Plotting the hidden weight visualizations
        plt.figure()
        for k in range(50):
            plt.subplot(5, 10, k + 1)
            plt.imshow(low_lambda_hidden[k].T, cmap='gray')
            plt.axis('off')
        plt.savefig('Q1_D2_L.png', bbox_inches='tight')
        plt.show()


        # Lambda Medium
        medium_lambda = 0.0005
        medium_lambda_train = network_solver(data_in, h_neuron_num, medium_lambda, rho, beta)
        med_lambda_hidden = hidden_layer_features(50, medium_lambda_train)

        # Plotting Lambda Med Hidden Weight Visualizations
        # Plotting the hidden weight visualizations
        plt.figure()
        for k in range(50):
            plt.subplot(5, 10, k + 1)
            plt.imshow(med_lambda_hidden[k].T, cmap='gray')
            plt.axis('off')
        plt.savefig('Q1_D2_M.png', bbox_inches='tight')
        plt.show()

        # Lambda High
        high_lambda = 0.001
        high_lambda_train = network_solver(data_in, h_neuron_num, high_lambda, rho, beta)
        high_lambda_hidden = hidden_layer_features(50, high_lambda_train)

        # Plotting Lambda High Hidden Weight Visualizations
        # Plotting the hidden weight visualizations
        plt.figure()
        for k in range(50):
            plt.subplot(5, 10, k + 1)
            plt.imshow(high_lambda_hidden[k].T, cmap='gray')
            plt.axis('off')
        plt.savefig('Q1_D2_H.png', bbox_inches='tight')
        plt.show()


    elif question == "2":
        print("Question 2 is a Jupyter Notebook Assignment")

    #*****************************************************************************************************************

    # Question 3
    elif question == "3":

        print('Question:3')

        # Retrieving the training Data from the h5 file as numpy arrays
        hf = h5py.File('assign3_data3.h5', 'r')
        ls = list(hf.keys())
        print('List of datasets in this file ', ls)
        # Obtaining the data from the h5 file
        trX = np.array(hf.get('trX'))  # (3000, 150, 3)
        trY = np.array(hf.get('trY'))  # (3000, 6)
        tstX = np.array(hf.get('tstX'))  # (600, 150, 3)
        tstY = np.array(hf.get('tstY'))  # (600, 6)

        # Part A
        print('Question:3:A')
        # RNN Network Initializations

        # Number of hidden units
        hidden_rnn = 128
        batch_size = 32
        # X-Y-Z plane data
        num_features = trX.shape[2]
        lrn = 0.1
        # Time Series Length
        time_steps = trX.shape[1] # 150 time steps for each sample
        # Max Epochs 50
        epochs = 30
        momentum = 0.85

        # MLP Network Parameters Initializations
        # MLP takes hidden state of rnn as its input
        # MLP input layer shape = (32,128)
        # Output shape = (32,6)
        hidden_layer_num = 1
        mlp_in = 32
        output_size = 6
        # Number of hidden neurons in MLP hidden layer
        hidden_mlp = 24
        # Weight and Bias Initializations for RNN with Xavier Initialization
        # RNN weights xavier limit
        xav_x = np.sqrt(6 / (batch_size + hidden_rnn))
        xav_a = np.sqrt(6 / (hidden_rnn + hidden_rnn))
        # RNN weights and bias initialization
        # a = g(W_a.a_prev + W_x.sample + bias)
        num_features = trX.shape[2] # = 3
        weights_x = np.random.uniform(-xav_x, xav_x, size=(3, 128))
        bias_x = np.random.uniform(-xav_x, xav_x, size=(1, 128))
        weights_a = np.random.uniform(-xav_a, xav_a, size=(128, 128))
        # Hidden_State is the output of the RNN and is of shape (32,128)
        # MLP takes hidden state of rnn as its input
        # MLP input layer shape = (32,128) = RNN output layer shape
        # Output shape = (32,6)
        # MLP weights and bias initialization
        hidden_layer_num = 1
        mlp_in = 32
        output_size = 6
        # hidden neurons in mlp's hidden layer
        hidden_mlp = 24
        # Xavier Limit for MLP hidden layer
        xav_mlp_hid = np.sqrt(6 / (batch_size + hidden_mlp))

        # weights and bias initializations on MLP hidden layer
        weights_1 = np.random.uniform(-xav_mlp_hid, xav_mlp_hid, size=(128, hidden_mlp))
        bias_1 = np.random.uniform(-xav_mlp_hid, xav_mlp_hid, size=(1, hidden_mlp))
        # weights and bias initializations on MLP output layer
        # Xavier Limit for MLP hidden layer
        xav_mlp_out = np.sqrt(6 / (batch_size + output_size))
        weights_2 = np.random.uniform(-xav_mlp_out, xav_mlp_out, size=(hidden_mlp, 6))
        bias_2 = np.random.uniform(-xav_mlp_out, xav_mlp_out, size=(1, 6))

        # Obtaining the Validation Data Set
        den_d = trX
        den_l = trY
        np.random.seed(12)
        # Shuffle trX data and trY labels to create validation set from 10% of the samples
        np.random.shuffle(den_d)
        np.random.shuffle(den_l)
        # 2700 samples in training set and 300 samples in validation set
        val_data = den_d[0:300][:][:]
        # print(val_data.shape) = # (300, 150, 3)
        train_data = den_d[300:3000][:][:]
        # print(train_data.shape)= (2700, 150, 3)
        val_label = den_l[0:300][:] # Shape (300, 150, 3)
        train_label = den_l[300:3000][:] # Shape (2700, 150, 3)

        # Reloading the unshuffled data
        trX = np.array(hf.get('trX'))  # (3000, 150, 3)
        trY = np.array(hf.get('trY'))  # (3000, 6)
        tstX = np.array(hf.get('tstX'))  # (600, 150, 3)
        tstY = np.array(hf.get('tstY'))  # (600, 6)

        # Keep parameters in a vector W_params
        W_params = weights_x, bias_x, weights_a, weights_1, bias_1, weights_2, bias_2

        # With the defined functions network is trained
        train_acc = list()
        test_acc = list()
        val_loss = list()
        W_enc = weights_x, bias_x, weights_a, weights_1, bias_1, weights_2, bias_2
        mom = 0.85
        lrn = 0.1
        batch_size = 32
        #  Training the Overall Network
        small_ep = 20
        for epoch in range(small_ep):
            tot_batch = int(math.floor(train_data.shape[0] / batch_size))
            wa_prev, wx_prev, bx_prev, w1_prev, w2_prev, b1_prev, b2_prev = 0, 0, 0, 0, 0, 0, 0
            deck = np.random.permutation(len(train_data))
            shuffled_x = train_data[deck]
            shuffled_y = train_label[deck]
            for bat in range(tot_batch):
                data_in = shuffled_x[bat * batch_size:bat * batch_size + batch_size]
                label_actual = shuffled_y[bat * batch_size:bat * batch_size + batch_size]

                # Forwarding the data through time
                # data_cache = inp_state, hidden_state_rnn, a1_state, out_state
                data_cache = forward_pass(data_in, W_enc)
                _, _, _, activ = data_cache
                # grad_cache = [dW_x,dB_x,dW_a, dW_1, dB_1,dW_2,dB_2]
                grad_cache = backstreets_back(data_cache, label_actual, W_enc)

                # Updates
                wx_up = lrn * grad_cache[0] / batch_size + (mom * wx_prev)
                weights_x -= lrn * grad_cache[0] / batch_size + (mom * wx_prev)
                wx_prev = wx_up

                bias_x -= lrn * grad_cache[1] / batch_size + (mom * bx_prev)
                bx_prev = lrn * grad_cache[1] / batch_size + (mom * bx_prev)

                weights_a -= lrn * grad_cache[2] / batch_size + (mom * wa_prev)
                wa_prev = lrn * grad_cache[2] / batch_size + (mom * wa_prev)

                weights_1 -= lrn * grad_cache[3] / batch_size + (mom * w1_prev)
                w1_prev = lrn * grad_cache[3] / batch_size + (mom * w1_prev)

                bias_1 -= lrn * grad_cache[4] / batch_size + (mom * b1_prev)
                b1_prev = lrn * grad_cache[4] / batch_size + (mom * b1_prev)

                weights_2 -= lrn * grad_cache[5] / batch_size + (mom * w2_prev)
                w2_prev = lrn * grad_cache[5] / batch_size + (mom * w2_prev)

                bias_2 -= lrn * grad_cache[6] / batch_size + (mom * b2_prev)
                b2_prev = lrn * grad_cache[6] / batch_size + (mom * b2_prev)

            W_upd = weights_x, bias_x, weights_a, weights_1, bias_1, weights_2, bias_2
            # Train Accuracy
            _, _, _, activ = data_cache
            last = len(activ)
            train_mec = acc_check(predict_out(activ[149]), label_actual)
            test_acc.append(train_mec)
            # Test Accuracy
            test_met = accuracy_test(tstX, tstY, W_upd)
            test_acc.append(test_met)

            valid_L = val_cross_ent(val_data, val_label, W_upd)
            val_loss.append(valid_L)
            # Stopping the Algorithm on cross_validation
            print('Epoch:' + str(epoch) + ' Validation Loss:' + str(valid_L))
            if ep > 5:
                if valid_L > val_loss[int(epoch) - 1]:
                    print('Algorithm Stopped on Epoch ' + str(epoch))
                    break


# Functions used in order
# Question 1 Functions
# Weight and Bias initialization interval function that computes w_0
def Q1_weight_bias_interv(Lpre,Lpost):

    interv = np.sqrt(6/(Lpre + Lpost))
    return interv

# Activation Function chosen sigmoid to keep output in [0,1] interval
def sigmoid_act(arr):

    return 1/(1+ np.exp(-arr))
# Derivative of sigmoid function
def sigmoid_derv(activ_hid):

    return activ_hid*(1-activ_hid)


# In order to obtain the cost function we first need to obtain rho_hat_j which represents a vector that
# contains average activation of hidden layer neuron j for all hidden neurons
# Assuming sigmoid activation since we expect data to be btw 0,1
def sparsity(activation_hidden, sample_size):
    # Summing and obtaining the overage for each of 64 hidden layer neuron activations
    sum_1 = np.sum(activation_hidden, axis=0)
    rho_hat = sum_1/sample_size
    return rho_hat


# Calculates Kullback-Leibler Divergence to penalize the network for large deviations from rho
def KL_divergence(rho, rho_hidden):
    KL_penalty = np.sum((rho*np.log(rho/rho_hidden)) + ((1-rho)*np.log((1-rho)/(1-rho_hidden))))
    return KL_penalty

# Obtains the partial derivative of the KL divergence
def KL_derivative(rho, rho_hidden,beta):
    KL_der = beta * (-(rho/rho_hidden) + ((1-rho)/(1-rho_hidden)))
    return KL_der


# Overall Auto Encoder Cost function that calculates the cost and its partial derivatives
def aeCost(W_enc, data, parameters):
    sample_N = data.shape[0]
    W_1, W_2, B_1, B_2 = W_enc
    L_in_ae, L_hid_ae, half_life, beta_ae, rho_ae = parameters
    # Forwarding the inputs
    linear_1 = np.dot(data,W_1) + B_1
    active_1 = sigmoid_act(linear_1)
    linear_2 = np.dot(active_1,W_2) + B_2
    active_2 = sigmoid_act(linear_2)
    prediction = active_2
    # Average Square Error between desired output and network output
    sum_sqr_error = np.sum(np.square(data-prediction))
    mse = sum_sqr_error/(2*sample_N)
    # Obtaining aver age activation of hidden neurons with the rho_hat
    rho_hid = sparsity(active_1,sample_N)
    # Obtaining KL Divergence
    divergent = beta_ae * KL_divergence(rho_ae,rho_hid)
    # Tykhonov Regularization
    Ty_reg = half_life/2 * (np.sum(np.square(W_1)) + np.sum(np.square(W_2)))
    # Overall Cost Function
    # AECost = mse + Tykonov Regularization + KL_div
    ae_cost = mse + Ty_reg + divergent
    error_grad = prediction - data

    # Backpropagation
    # On the Output Layer
    # Partial Derivative of Cost wrt to Output Layer Activation
    dA_2 = error_grad
    # Partial Derivative of Cost wrt to Output Layer Linear Action Potential
    dZ_2 = dA_2 * sigmoid_derv(active_2)
    # Partial Derivative of Regularization Term with respect to W_2
    dRg_2 = half_life*W_2
    # Gradient of W2
    dW_2 = np.dot(active_1.T,dZ_2) + dRg_2
    grad_W2 = dW_2 / sample_N
    # Gradient of B2
    dB_2 = np.sum(dZ_2, axis=0, keepdims=True)
    grad_B2 = dB_2 / sample_N

    # On the Hidden Layer
    # Partial Derivative of KL Divergence wrt to Hidden Layer Activation
    dKL_1 = KL_derivative(rho_ae,rho_hid,beta_ae)
    # Partial Derivative of Cost wrt to Hidden Layer Activation
    dA_1 = np.dot(dZ_2,W_2.T) + (beta_ae * dKL_1)
    # Partial Derivative of Cost wrt to Hidden Layer Linear Action Potential
    dZ_1 = dA_1 * sigmoid_derv(active_1)
    # Partial Derivative of Regularization Term with respect to W_1
    dRg_1 = half_life * W_1

    # Gradient of W1
    dW_1 = np.dot(data.T,dZ_1) + dRg_1
    grad_W1 = dW_1/sample_N
    # Gradient of B1
    dB_1 = np.sum(dZ_1, axis=0, keepdims=True)
    grad_B1 = dB_1/sample_N

    return ae_cost,grad_W1,grad_B1,grad_W2,grad_B2

# Obtaining the predictions of the trained network for given data sample
def feature_ext(image, W_trained, params):
    W_1, W_2, B_1, B_2 = W_trained
    L_in_ae, L_hid_ae, half_life, beta_ae, rho_ae = params
    # Extracting Features of the Samples
    linear_1 = np.dot(image, W_1) + B_1
    active_1 = sigmoid_act(linear_1)
    linear_2 = np.dot(active_1, W_2) + B_2
    active_2 = sigmoid_act(linear_2)
    prediction = active_2
    return prediction

# For Question 1 Part D
# Creating a general network function to implement stochastic gradient descent learning algorithm
# and obtain trained hidden Weights for visualization
def network_solver(data, L_hidden, lam_param, rho_param,beta_param):
    # Network Initializations
    # # Normalized data is reshaped to (10240,256)
    # data_in = normal_data.reshape(10240, 256)
    # Network parameters
    lamb_param = lam_param
    rho = rho_param
    beta = beta_param
    # Data is of size L_in * N where L_in = 16*16 = 256 and N = 10240
    L_in = data.shape[1]
    L_hid = L_hidden
    # Number of input neurons
    L_pre_1 = data.shape[0]  # N=10240
    sample = L_pre_1
    # Number of Neurons on the input to hidden layer
    L_post_1 = L_hid
    # Initializing weight and biases of the first layer
    w_0 = Q1_weight_bias_interv(L_pre_1, L_post_1)
    weights_1 = np.random.uniform(-1 * w_0, w_0, (256, L_hid))
    bias_1 = np.random.uniform(-1 * w_0, w_0, (1, L_hid))
    # Initializing weight and biases of the hidden to output layer
    L_pre_2 = L_post_1
    L_post_2 = L_pre_1
    # The same w_0 value is obtained as the previous layer
    weights_2 = np.random.uniform(-1 * w_0, w_0, (L_hid, 256))
    bias_2 = np.random.uniform(-1 * w_0, w_0, (1, 256))

    # Creating the W_e vector that contains all weights and biases
    W_e = weights_1, weights_2, bias_1, bias_2
    # Creating a params structure that contains parameters of the network
    params = L_in, L_hid, lamb_param, beta, rho

    # Unsupervised Learning
    lrn = 1
    epochs = 1000
    error_metrics = list()
    for epoch in range(epochs):
        cost, dW1, dB1, dW2, dB2 = aeCost(W_e, data, params)
        # Updating Weights and Biases
        weights_2 -= dW2 * lrn
        bias_2 -= dB2 * lrn
        weights_1 -= dW1 * lrn
        bias_1 -= dB1 * lrn
        # Cost over Epochs
        error_metrics.append(cost)
        if int(epoch) % 200 == 0:
            print('#Epoch:' + str(epoch) + '--> Cost = ' + str(cost))
    W_train =  weights_1

    return W_train

# Function to convert hidden weights to imshow format
def hidden_layer_features(hidden_neuron,weights_hidden):
    w_hid = weights_hidden.T
    hidden_features = w_hid.reshape(hidden_neuron,16,16)
    return hidden_features

######################################################################
# Question 3 Functions

# Hyperbolic tangent activation function
def tanh(arr):
    output = (np.exp(arr) - np.exp(-arr)) / (np.exp(arr) + np.exp(-arr))
    return output
# Derivative of the tanh function
def tanh_derivative(active_out):
    derivative = 1-(active_out**2)
    return derivative

# MLP activation Softmax
def softmax_act(arr):
    e_z = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
    activated = e_z / np.sum(e_z, axis=-1, keepdims=True)
    return activated


# Derivative of the softmax function
def softmax_derv(activ_out):
    grad_soft = activ_out * (1 - activ_out)
    return grad_soft

# MLP hidden Activation function chosen as ReLu
def ReLu(lin):
    lin[lin <= 0] = 0
    return lin


def ReLu_derv(lin):
    lin[lin <= 0] = 0
    lin[lin > 0] = 1

    return lin


# Classify the predictions based on the highest probability
def predict_out(activ_output):
    row = activ_output.shape[0]
    col = activ_output.shape[1]
    adjust = np.argmax(activ_output,axis=1)
    prediction_array = np.zeros((row,col))
    for i in range(row):
        prediction_array[i][adjust[i]] = 1

    return prediction_array

# Categorical Cross entropy Loss function
def cat_cross_entropy(predict,label):
    samples = predict.shape[0]
    prediction = np.clip(predict, 1e-15, 1 - 1e-15)
    cost = -np.sum(label * np.log(prediction + 1e-15))
    cr_out = cost/samples
    return cr_out

# Derivative of the Loss function
def cross_derv(predict,label):
    predictions = np.clip(predict, 1e-15, 1 - 1e-15)
    der_ce = - (label/predictions)
    return der_ce


# General Function for forwarding the inputs through the network and caching relevant data
# General Function for forwarding the inputs through the network and caching relevant data
def forward_pass(batch_data, W_encode):
    # Keeping track of time
    hidden_state_rnn = dict()
    a1_state = dict()
    inp_state = dict()
    out_state = dict()
    time_length = 150
    # Initial Hidden State
    sample = batch_data.shape[0]
    a_prev = np.zeros((sample, 128))
    hidden_state_rnn[-1] = a_prev
    #  Forwarding input through the network

    w_x, b_x, w_a, w_1, b_1, w_2, b_2 = W_encode
    for t in range(150):
        sample_data = batch_data[:, t, :]
        inp_state[t] = sample_data
        # RNN architecture Forwarding
        # rnn_linear = batch_size,128
        rnn_linear = (np.dot(sample_data, w_x) + np.dot(hidden_state_rnn[t - 1], w_a)) + b_x
        # a = batch_size,128
        activ_current = tanh(rnn_linear)
        hidden_state_rnn[t] = activ_current

        #  Moving to MLP architecture
        # lin_1 = (batch_size,hidden_mlp_size) = (32,12)
        lin_1 = np.dot(activ_current, w_1) + b_1
        # activ_1 = (batch_size,hidden_mlp_size) = (32,12)
        activ_1 = ReLu(lin_1)
        a1_state[t] = activ_1
        # lin_2 = (batch_size,6)
        lin_2 = np.dot(activ_1, w_2) + b_2
        # act_2 = (batch_size,6)
        activ_2 = softmax_act(lin_2)
        out_state[t] = activ_2
        #  storing data for backprop through time
    cache = inp_state, hidden_state_rnn, a1_state, out_state
    return cache




# Accuracy Checking
def acc_check(data, label):
    size = data.shape[0]
    max_dat = np.argmax(data, axis=1)
    max_lab = np.argmax(label, axis=1)
    count = 0
    for k in range(size):
        if max_dat[k] == max_lab[k]:
            count = count + 1

    return count * 100 / size


def accuracy_test(test_x, test_y, W_updated):
    cache_test = forward_pass(test_x, W_updated)
    _, _, _, out_test = cache_test
    last = len(out_test)
    prediction_test = predict_out(out_test[149])
    percantage_acc = acc_check(prediction_test, test_y)
    return percantage_acc


# Cross entropy loss on validation to stop the algorithm
def val_cross_ent(val_x, val_y, W_updated):
    cache_val = forward_pass(val_x, W_updated)
    _, _, _, _, out_val = cache_val
    last = len(out_val)
    validation_loss = cat_cross_entropy(out_val[149], val_y)
    return validation_loss


# Backpropagation through time algorithm implemented with cached data from forward pass
def backstreets_back(cache, labels, W_enc):
    # cache = inp_state, hidden_state_rnn, a1_state, out_state
    inp_state, hidden_state_rnn, a1_state, out_state = cache
    W_x, B_x, W_a, W_1, B_1, W_2, B_2 = W_enc
    # Initializing  gradient arrays for w and b
    dW_x, dB_x, dW_a = np.zeros_like(W_x), np.zeros_like(B_x), np.zeros_like(W_a)
    dW_1, dB_1 = np.zeros_like(W_1), np.zeros_like(B_1)
    dW_2, dB_2 = np.zeros_like(W_2), np.zeros_like(B_2)

    # On the MLP output Layer
    last_index = len(out_state)
    last_act = len(a1_state)
    error_grad = cross_derv(out_state[149], labels)
    dZ_2 = error_grad * softmax_derv(out_state[149])
    dW_2 = np.dot(a1_state[149].T, dZ_2)
    dB_2 = np.sum(dZ_2, axis=0, keepdims=True)

    # On the MLP Hidden Layer

    dA_1 = np.dot(dZ_2, W_2.T)
    dZ_1 = dA_1 * ReLu_derv(a1_state[149])
    dW_1 = np.dot(hidden_state_rnn[149].T, dZ_1)
    dB_1 = np.sum(dZ_1, axis=0, keepdims=True)

    # On RNN
    dA_xn = np.zeros_like(hidden_state_rnn[0])
    for t in reversed(range(1, 150)):
        dA_x = np.dot(dZ_1, W_1.T) + dA_xn
        dZ_x = tanh_derivative(hidden_state_rnn[t]) * dA_x

        dW_x += np.dot(inp_state[t].T, dZ_x)
        dW_a += np.dot(hidden_state_rnn[t - 1].T, dZ_x)
        dB_x += np.sum(dZ_x, axis=0, keepdims=True)

        dA_xn = np.dot(dZ_x, W_a.T)

    grad_clip = [dW_x, dB_x, dW_a, dW_1, dB_1, dW_2, dB_2]
    # Gradient Clipping
    for k in grad_clip:
        np.clip(k, -10, 10, out=k)

    grad_cache = [dW_x, dB_x, dW_a, dW_1, dB_1, dW_2, dB_2]
    alright = grad_cache
    return alright

arman_budunoglu_21602635_hw3(question)