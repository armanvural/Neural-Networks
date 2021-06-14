import h5py
import numpy as np
import numpy.ma as ma
import sys
import matplotlib.pyplot as plt
import random
import math


question = input("Enter Relevant Question Number to Display its Respective Output \n")


def arman_budunoglu_21602635_hw2(question):

    if question == "1":

        print("Question 1: ")

        # Retrieving the training Data from the h5 file as numpy arrays
        hf = h5py.File('assign2_data1.h5', 'r')
        ls = list(hf.keys())

        print('List of datasets in this file ', ls)
        # Getting the labels and the samples from the training data
        train_images = np.array(hf.get('trainims'))
        train_labels = np.array(hf.get('trainlbls'))
        # Getting the labels and the samples from the test data
        test_labels = np.array(hf.get('testlbls'))
        test_images = np.array(hf.get('testims'))

        # Checking the shapes of the data arrays
        # #train_ims shape = (1900,32,32)
        # print("Training Images Shape = " + str(train_images.shape))
        # # train_lbls shape = (1900,)
        # print("Training Labels Shape = " + str(train_labels.shape))
        # # test_ims shape = (1000,32,32)
        # print("Testing Images Shape = " + str(test_images.shape))
        # # test_lbls shape = (1000,)
        # print("Testing Labels Shape = " + str(test_images.shape))
        # 950 "0" labels 950 "1" labels
        # print(train_labels[940:960])
        # Plotting samples btw 940-960
        # plt.figure()
        # usher = 0
        # for i in range(940,960):
        #     usher += 1
        #     image = train_images[i]
        #     plt.subplot(4,5,usher)
        #     plt.imshow(image)
        # plt.show()
        # Question 1 Part A

        print('Question 1: Part A')
        # Reshaping training and test images and labels
        train_images = train_images.reshape(1900, 1024)
        train_labels = train_labels.reshape(1900, 1)
        test_images = test_images.reshape(1000, 1024)
        test_labels = test_labels.reshape(1000, 1)
        # Manipulating the labels to intersect with the activation function output
        train_labels[train_labels == 0] = -1
        test_labels[test_labels == 0] = -1
        # Normalizing image data
        train_images = train_images / 255
        test_images = test_images / 255

        # Initializing network parameters
        h_neuron = 24
        batch_size = 38
        # Using Xavier Initialization for weights and biases std = sqrt(2/input+output)
        std_1 = np.sqrt(2 / (batch_size + h_neuron))
        std_2 = np.sqrt(2 / (1 + h_neuron))
        weights_1 = np.random.normal(0, std_1, size=(1024, h_neuron))
        weights_2 = np.random.normal(0, std_2, size=(h_neuron, 1))
        bias_1 = np.zeros(h_neuron).reshape(1, h_neuron)
        bias_2 = np.zeros(1).reshape(1, 1)
        epochs = 250
        learning_rate = 0.1
        mce_train = list()
        mse_train = list()
        mse_test = list()
        mce_test = list()

        # Training the network in part A
        for test in range(epochs):
            tot = int(math.floor(1900 / batch_size))
            deck = np.random.permutation(len(train_images))
            shuffled_x = train_images[deck]
            shuffled_y = train_labels[deck]
            for i in range(tot):
                data_batch = shuffled_x[i * batch_size:i * batch_size + batch_size]
                label_batch = shuffled_y[i * batch_size:i * batch_size + batch_size]

                linear_action_pot_1 = np.dot(data_batch, weights_1) + bias_1
                active_1 = tanh(linear_action_pot_1)

                linear_action_pot_2 = np.dot(active_1, weights_2) + bias_2
                active_2 = tanh(linear_action_pot_2)

                prediction = active_2

                cost = np.square(prediction - label_batch)
                error = prediction - label_batch

                # Backpropagation
                # Output Layer
                # Weight_2 Gradient for output layer dC/dW2 = dL2/dW2 * dA2/dL2 * dC/dA2
                dL2_W2 = active_1
                dA2_L2 = tanh_derivative(active_2)
                dC_A2 = error
                # dZ2 = dC_A2 * dA2_L2
                dZ2 = dA2_L2 * dC_A2
                # Gradient of W2
                grad_W2 = np.dot(dL2_W2.T, dZ2)
                # Bias_2 Gradient dC/dB2 = dL2/dB2 * dA2/dL2 * dC/dA2
                # dL2_B2 = 1
                # Gradient of B2
                grad_B2 = np.sum(dZ2, axis=0, keepdims=True)
                # Updating the weights and biases for Output Layer
                weights_2 -= learning_rate / batch_size * grad_W2
                bias_2 -= learning_rate / batch_size * grad_B2

                # Layer 1
                # Weight_1 Gradient for Hidden layer dC/dW1 = dL2/dW1 * dA2/dL2 * dC/dA2
                # dL2/dW1 = dL1/dW1 * dA1/dL1 * dL2/dA1
                dL1_W1 = data_batch
                dA1_L1 = tanh_derivative(active_1)
                dL2_A1 = weights_2
                dZ1 = np.dot(dZ2, dL2_A1.T) * dA1_L1
                # Gradient of W1
                grad_W1 = np.dot(dL1_W1.T, dZ1)
                # Bias_2 Gradient dC/dB2 = dL2/dB2 * dA2/dL2 * dC/dA2
                dL2_B1 = 1
                # Gradient of B1
                grad_B1 = np.sum(dZ1, axis=0, keepdims=True)
                # Updating the weights and biases for Output Layer
                weights_1 -= learning_rate / batch_size * grad_W1
                bias_1 -= learning_rate / batch_size * grad_B1

            # Obtaining the error metrics for the training data set
            train_score = error_metrics(train_images, weights_1, bias_1, weights_2, bias_2, train_labels, 1900, 2,
                                        'nan', 'nan')
            mse_training = train_score['MSE_val']
            mce_training = train_score['MCE_val']
            mse_train.append(mse_training)
            mce_train.append(mce_training)
            # Obtaining the error metrics for the test data set
            test_score = error_metrics(test_images, weights_1, bias_1, weights_2, bias_2, test_labels, 1000, 2, 'nan', 'nan')
            mse_testing = test_score['MSE_val']
            mce_testing = test_score['MCE_val']
            mse_test.append(mse_testing)
            mce_test.append(mce_testing)

        # Plotting Error metrics for training and test results 2 layer network
        plt.plot(mse_train)
        plt.title('MSE_Training vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Train')
        plt.savefig(" PART:A: 2 Layer MSE Train.png")
        plt.show()

        plt.plot(mce_train)
        plt.title('MCE_Training vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MCE Train')
        plt.savefig("PART:A: 2 Layer MCE Train.png")
        plt.show()

        plt.plot(mse_test)
        plt.title('MSE_Test vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Test')
        plt.savefig("PART:A: 2 Layer MSE Test.png")
        plt.show()

        plt.plot(mce_test)
        plt.title('MCE_Test vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MCE Test')
        plt.savefig("PART:A: 2 Layer MCE TEST.png")
        plt.show()


        print('Question 1: Part C')

        # Training a network with lower hidden layer neurons
        low_neuron = 7
        # Training and obtaining error metrics for the n_low
        smaller_neuron_metrics = neuron_size(low_neuron,train_images,train_labels,test_images,test_labels,250,38)
        low_train_mse = smaller_neuron_metrics['MSE_rain']
        low_train_mce = smaller_neuron_metrics['MCE_rain']
        low_test_mse = smaller_neuron_metrics['MSE_tes']
        low_test_mce = smaller_neuron_metrics['MCE_tes']
        # Training a network with higher hidden layer neurons
        high_neuron = 380
        # Training and obtaining error metrics for the n_high
        higher_neuron_metrics = neuron_size(high_neuron, train_images, train_labels, test_images, test_labels, 250, 38)
        high_train_mse = higher_neuron_metrics['MSE_rain']
        high_train_mce = higher_neuron_metrics['MCE_rain']
        high_test_mse = higher_neuron_metrics['MSE_tes']
        high_test_mce = higher_neuron_metrics['MCE_tes']

        # Plotting the error metrics for n_low,n_high and n*
        # Training MSE
        plt.figure()
        plt.plot(high_train_mse)
        plt.plot(low_train_mse)
        plt.plot(mse_train)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Value")
        plt.title('Training MSE for Neuron Units')
        plt.legend(['N_high =' + str(high_neuron), 'N_low =' + str(low_neuron),'N^* =' + str(h_neuron)])
        plt.savefig("PART:C: Train MSE's Neuron Size MSE's.png")
        plt.show()
        # Training MCE
        plt.figure()
        plt.plot(high_train_mce)
        plt.plot(low_train_mce)
        plt.plot(mce_train)
        plt.xlabel("Epoch")
        plt.ylabel("MCE Value")
        plt.title('Training MCE for Neuron Units')
        plt.legend(['N_high =' + str(high_neuron), 'N_low =' + str(low_neuron),'N^* =' + str(h_neuron)])
        plt.savefig("PART:C:Train MCE's Neuron Size MCE's.png")
        plt.show()
        # Test MSE
        plt.figure()
        plt.plot(high_test_mse)
        plt.plot(low_test_mse)
        plt.plot(mse_test)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Value")
        plt.title('Testing MSE for Neuron Units')
        plt.legend(['N_high =' + str(high_neuron), 'N_low =' + str(low_neuron),'N^* =' + str(h_neuron)])
        plt.savefig("PART:C:Test MSE's Neuron Size MSE's.png")
        plt.show()
        # Test MCE
        plt.figure()
        plt.plot(high_test_mce)
        plt.plot(low_test_mce)
        plt.plot(mce_test)
        plt.xlabel("Epoch")
        plt.ylabel("MCE Value")
        plt.title('Training MCE for Neuron Units')
        plt.legend(['N_high =' + str(high_neuron), 'N_low =' + str(low_neuron), 'N^* =' + str(h_neuron)])
        plt.savefig("PART:C:Test MCE's Neuron Size MCE's.png")
        plt.show()



        print('Question 1: Part D')
        # Three layer network parameter initializations
        batch_3 = 38
        hidden_1 = 42
        hidden_2 = 12
        xav_1 = np.sqrt(2 / (batch_3 + hidden_1))
        xav_2 = np.sqrt(2 / (hidden_1 + hidden_2))
        xav_3 = np.sqrt(2 / (1 + hidden_2))
        W1 = np.random.normal(0, xav_1, size=(1024, hidden_1))
        W2 = np.random.normal(0, xav_2, size=(hidden_1, hidden_2))
        W3 = np.random.normal(0, xav_3, size=(hidden_2, 1))
        B1 = np.zeros(hidden_1).reshape(1, hidden_1)
        B2 = np.zeros(hidden_2).reshape(1, hidden_2)
        B3 = np.zeros(1).reshape(1, 1)
        # Training and Testing the Network
        three_layer_performance = three_layer_network(train_images, train_labels, W1, B1, W2, B2, W3, B3, batch_3, 250,
                                                      0.1, 24, test_images, test_labels,0)
        mean_square_train_3 = three_layer_performance['train_MSE']
        mean_class_train_3 = three_layer_performance['train_MCE']
        mean_square_test_3 = three_layer_performance['test_MSE']
        mean_class_test_3 = three_layer_performance['test_MCE']

        # Plotting Error metrics for training and test results on 3 layer network
        plt.plot(mean_square_train_3, 'k')
        plt.title('3 Layer MSE over Training vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE over Training')
        plt.savefig("PARTD:3 Layer MSE Train.png")
        plt.show()

        plt.plot(mean_class_train_3, 'k')
        plt.title('3 Layer MCE over Training vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MCE Training')
        plt.savefig("PARTD:3 Layer MCE Train.png")
        plt.show()

        plt.plot(mean_square_test_3, 'r')
        plt.title('3 Layer MSE over Test vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('3 MSE Test')
        plt.savefig("PARTD:3 Layer MSE Test.png")
        plt.show()

        plt.plot(mean_class_test_3, 'r')
        plt.title('3 Layer MCE Test vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Classification Error on Testing')
        plt.savefig("PARTD:3 Layer MCE Test.png")
        plt.show()

        # Comparison of 3 layer Network to 2 Layer

        # Plotting Error metrics for training and test results on 3 layer network
        plt.plot(mean_square_train_3, 'k')
        plt.plot(mse_train)
        plt.title('2vs3 MSE over Training vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE over Training')
        plt.legend(['Hidden Layer =' + str(2), 'Hidden Layer =' + str(1)])
        plt.savefig("PARTD:2vs3 Layer MSE Train.png")
        plt.show()

        plt.plot(mean_class_train_3, 'k')
        plt.plot(mce_train)
        plt.title('2vs3 Layer MCE over Training vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MCE Training')
        plt.legend(['Hidden Layer =' + str(2), 'Hidden Layer =' + str(1)])
        plt.savefig("PARTD:2vs3 Layer MCE Train.png")
        plt.show()

        plt.plot(mean_square_test_3, 'r')
        plt.plot(mse_test)
        plt.title('2vs3 Layer MSE over Test vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Test')
        plt.legend(['Hidden Layer =' + str(2), 'Hidden Layer =' + str(1)])
        plt.savefig("PARTD:2vs3 Layer MSE Test.png")
        plt.show()

        plt.plot(mean_class_test_3, 'r')
        plt.plot(mce_test)
        plt.title('3 Layer MCE Test vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Classification Error on Testing')
        plt.legend(['Hidden Layer =' + str(2), 'Hidden Layer =' + str(1)])
        plt.savefig("PARTD:2vs3 MCE_test MCE Test.png")
        plt.show()

        print('Question 1: Part E')
        # Reinitializing the network for momentum terms
        batch_3 = 38
        hidden_1 = 42
        hidden_2 = 12
        xav_1 = np.sqrt(2 / (batch_3 + hidden_1))
        xav_2 = np.sqrt(2 / (hidden_1 + hidden_2))
        xav_3 = np.sqrt(2 / (1 + hidden_2))
        W1 = np.random.normal(0, xav_1, size=(1024, hidden_1))
        W2 = np.random.normal(0, xav_2, size=(hidden_1, hidden_2))
        W3 = np.random.normal(0, xav_3, size=(hidden_2, 1))
        B1 = np.zeros(hidden_1).reshape(1, hidden_1)
        B2 = np.zeros(hidden_2).reshape(1, hidden_2)
        B3 = np.zeros(1).reshape(1, 1)
        momentum = 0.3
        momentum_metrics = three_layer_network(train_images, train_labels, W1, B1, W2, B2, W3, B3, batch_3, 250,
                                                      0.1, 24, test_images, test_labels,momentum)
        #Train MSE
        mse_mom_train = momentum_metrics['train_MSE']
        # Train MCE
        mce_mom_train = momentum_metrics['train_MCE']
        # Test MSE
        mse_mom_test = momentum_metrics['test_MSE']
        # Train MSE
        mce_mom_test = momentum_metrics['test_MCE']
        #Plotting error metrics with and without momentum for comparison
        # Train MSE
        plt.figure()
        plt.plot(mse_mom_train)
        plt.plot(mean_square_train_3)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Value")
        plt.title('Training MSE with/without Momentum')
        plt.legend(['Momentum =' + str(momentum), 'Momentum =' + str(0)])
        plt.savefig("PART:E: Train MSE's Momentum MSE's.png")
        plt.show()
        # Training MCE
        plt.figure()
        plt.plot(mce_mom_train)
        plt.plot(mean_class_train_3)
        plt.xlabel("Epoch")
        plt.ylabel("MCE Value")
        plt.title('Training MCE with/without Momentum')
        plt.legend(['Momentum =' + str(momentum), 'Momentum =' + str(0)])
        plt.savefig("PART:E:Train MCE's Momentum MCE's.png")
        plt.show()
        # Test MSE
        plt.figure()
        plt.plot(mse_mom_test)
        plt.plot(mean_square_test_3)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Value")
        plt.title('Testing MSE with/without Momentum')
        plt.legend(['Momentum =' + str(momentum), 'Momentum =' + str(0)])
        plt.savefig("PART:E:Test MSE's Momentum MSE's.png")
        plt.show()
        # Test MCE
        plt.figure()
        plt.plot(mce_mom_test)
        plt.plot(mean_class_test_3)
        plt.xlabel("Epoch")
        plt.ylabel("MCE Value")
        plt.title('Testing MCE with/without Momentum')
        plt.legend(['Momentum =' + str(momentum), 'Momentum =' + str(0)])
        plt.savefig("PART:E:Test MCE's Momentum MCE's.png")
        plt.show()





    elif question == "2":
        print("Question 2:")

        # Retrieving the training Data from the h5 file as numpy arrays
        hf = h5py.File('assign2_data2.h5', 'r')
        ls = list(hf.keys())
        print('List of datasets in this file ', ls)
        # Getting the labels and the samples from the training data
        train_d = np.array(hf.get('traind'))
        train_x = np.array(hf.get('trainx'))
        # Getting the labels and the samples from the test data
        test_d = np.array(hf.get('testd'))
        test_x = np.array(hf.get('testx'))
        # Getting validation data
        val_d = np.array(hf.get('vald'))
        val_x = np.array(hf.get('valx'))
        # Getting words
        words = np.array(hf.get('words'))

        # Checking the shapes of the data arrays
        #Traind Shape = (372500,)
        print("Traind Shape = " + str(train_d.shape))
        # Trainx Labels Shape = (372500, 3)
        print("Trainx Labels Shape = " + str(train_x.shape))
        # Testd  Shape = (46500,)
        print("Testd  Shape = " + str(test_d.shape))
        #Testx Shape = (46500, 3)
        print("Testx Shape = " + str(test_x.shape))
        #vald  Shape = (46500,)
        print("vald  Shape = " + str(val_d.shape))
        #valx Shape = (46500, 3)
        print("valx Shape = " + str(val_x.shape))
        # words Shape = (250,)
        print("words Shape = " + str(words.shape))

        # Obtain the one_hot encoded numpy arrays and store them to save time
        # 372500 words
        # 3 words form the trigram
        # 4th word is guessed and compared to label
        # reshaping data
        train_d = train_d.reshape(len(train_d), 1)
        test_d = test_d.reshape(len(test_d), 1)
        val_d = val_d.reshape(len(val_d), 1)

        print('Question:2:Part:A')
        # One Hot Encoding the data and labels and saving them to reduce runtime
        # encoded_train_x = np.sum(one_hot(train_x,train_d,0),axis=1)
        # np.save('train_x_enc.npy',encoded_train_x)
        # encoded_train_d = one_hot(train_x,train_d,1)
        # np.save('train_d_enc.npy',encoded_train_d)
        # # Testing
        # encoded_test_x = np.sum(one_hot(test_x,test_d,0),axis=1)
        # np.save('test_x_enc.npy',encoded_test_x)
        # encoded_test_d = one_hot(test_x,test_d,1)
        # np.save('test_d_enc.npy',encoded_test_d)
        # # Validation
        # encoded_val_x = np.sum(one_hot(val_x,val_d,0),axis=1)
        # np.save('val_x_enc.npy',encoded_val_x)
        # encoded_val_d = one_hot(val_x,val_d,1)
        # np.save('val_d_enc.npy',encoded_val_d)

        # Loading encoded one_hot data
        encoded_train_x = np.load('train_x_enc.npy')
        encoded_train_d = np.load('train_d_enc.npy')
        encoded_test_x = np.load('test_x_enc.npy')
        encoded_test_d = np.load('test_d_enc.npy')
        encoded_val_x = np.load('val_x_enc.npy')
        encoded_val_d = np.load('val_d_enc.npy')


        # Q2_metrics = {'TrainLoss': '', 'TrainAcc': '', 'TestLoss': '', 'TestAcc': '', 'ValLoss': '', 'ValAcc': '',
        #               'W_E': '', 'W_1': '', 'W_2': '', 'B_1': '', 'B_2': ''}

        # # Network with D,P = 8,64
        # D8_P64_metrics = Q2_part_A(8,64,0.5,0.85,40,encoded_train_x,encoded_train_d,encoded_test_x,encoded_test_d,encoded_val_x,encoded_val_d)
        # # Saving the updated weights and biases
        # case1_WE = D8_P64_metrics['W_E']
        # case1_W1 = D8_P64_metrics['W_1']
        # case1_W2 = D8_P64_metrics['W_2']
        # case1_B1 = D8_P64_metrics['B_1']
        # case1_B2 = D8_P64_metrics['B_2']
        # np.save('case1_WE', case1_WE)
        # np.save('case1_W1', case1_W1)
        # np.save('case1_W2', case1_W2)
        # np.save('case1_B1', case1_B1)
        # np.save('case1_B2', case1_B2)
        # #
        # # # case1_WE = np.load('case1_WE')
        # # case1_W1 = np.load('case1_W1')
        # # case1_W2 = np.load('case1_W2')
        # # case1_B1 = np.load('case1_B1')
        # # case1_B2 = np.load('case1_B2')
        #
        # Train Loss
        # plt.figure()
        # plt.plot(D8_P64_metrics['TrainLoss'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Training Loss')
        # plt.legend(['D,P = 8,64'])
        # plt.savefig("Q2:A:Train Loss.png")
        # plt.show()
        # # Train Acc
        # plt.figure()
        # plt.plot(D8_P64_metrics['TrainAcc'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Training Acc')
        # plt.legend(['D,P = 8,64'])
        # plt.savefig("Q2:A:Train ACc.png")
        # plt.show()
        # # Test Loss
        # plt.figure()
        # plt.plot(D8_P64_metrics['TestLoss'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Test Loss')
        # plt.legend(['D,P = 8,64'])
        # plt.savefig("Q2:A:Test Loss.png")
        # plt.show()
        # # Test Acc
        # plt.figure()
        # plt.plot(D8_P64_metrics['TestAcc'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Test Acc')
        # plt.legend(['D,P = 8,64'])
        # plt.savefig("Q2:A:Test Acc.png")
        # plt.show()
        # # Val Loss
        # plt.figure()
        # plt.plot(D8_P64_metrics['ValLoss'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Val Loss')
        # plt.legend(['D,P = 8,64'])
        # plt.savefig("Q2:A:Val Loss.png")
        # plt.show()
        # # Test Acc
        # plt.figure()
        # plt.plot(D8_P64_metrics['ValAcc'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Val Acc')
        # plt.legend(['D,P = 8,64'])
        # plt.savefig("Q2:A:Val Acc.png")
        # plt.show()
        #
        #
        #
        #
        #
        # # Network with D,P = 16,128
        # D16_P128_metrics = Q2_part_A(16, 128, 0.5, 0.85, 40, encoded_train_x, encoded_train_d, encoded_test_x,
        #                            encoded_test_d, encoded_val_x, encoded_val_d)
        #
        # # Saving the updated weights and biases
        # # case2_WE = D16_P128_metrics['W_E']
        # # case2_W1 = D16_P128_metrics['W_1']
        # # case2_W2 = D16_P128_metrics['W_2']
        # # case2_B1 = D16_P128_metrics['B_1']
        # # case2_B2 = D16_P128_metrics['B_2']
        # # np.save('case2_WE', case2_WE)
        # # np.save('case2_W1', case2_W1)
        # # np.save('case2_W2', case2_W2)
        # # np.save('case2_B1', case2_B1)
        # # np.save('case2_B2', case2_B2)
        #
        #
        # # np.save('case2_WE', case2_WE)
        # # np.save('case2_W1', case2_W1)
        # # np.save('case2_W2', case2_W2)
        # # np.save('case2_B1', case2_B1)
        # # np.save('case2_B2', case2_B2)
        #
        # # case2_WE = np.load('case2_WE')
        # # case2_W1 = np.load('case2_W1')
        # # case2_W2 = np.load('case2_W2')
        # # case2_B1 = np.load('case2_B1')
        # # case2_B2 = np.load('case2_B2')
        # #
        # # # Train Loss
        # plt.figure()
        # plt.plot(D16_P128_metrics['TrainLoss'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Training Loss')
        # plt.legend(['D,P = 16,128'])
        # plt.savefig("Q2:B:Train Loss.png")
        # plt.show()
        # # Train Acc
        # plt.figure()
        # plt.plot(D16_P128_metrics['TrainAcc'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Training Acc')
        # plt.legend(['D,P = 16,128'])
        # plt.savefig("Q2:B:Train ACc.png")
        # plt.show()
        # # Test Loss
        # plt.figure()
        # plt.plot(D16_P128_metrics['TestLoss'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Test Loss')
        # plt.legend(['D,P = 16,128'])
        # plt.savefig("Q2:B:Test Loss.png")
        # plt.show()
        # # Test Acc
        # plt.figure()
        # plt.plot(D16_P128_metrics['TestAcc'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Test Acc')
        # plt.legend(['D,P = 16,128'])
        # plt.savefig("Q2:B:Test Acc.png")
        # plt.show()
        # # Val Loss
        # plt.figure()
        # plt.plot(D16_P128_metrics['ValLoss'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Val Loss')
        # plt.legend(['D,P = 16,128'])
        # plt.savefig("Q2:B:Val Loss.png")
        # plt.show()
        # # Test Acc
        # plt.figure()
        # plt.plot(D16_P128_metrics['ValAcc'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Val Acc')
        # plt.legend(['D,P = 16,128'])
        # plt.savefig("Q2:B:Val Acc.png")
        # plt.show()
        # #
        # #
        # # # Network with D,P = 32,256
        # D32_P256_metrics = Q2_part_A(32, 256, 0.5, 0.85, 40, encoded_train_x, encoded_train_d, encoded_test_x,
        #                            encoded_test_d, encoded_val_x, encoded_val_d)
        # #
        # # # Saving the updated weights and biases
        # case3_WE = D32_P256_metrics['W_E']
        # case3_W1 = D32_P256_metrics['W_1']
        # case3_W2 = D32_P256_metrics['W_2']
        # case3_B1 = D32_P256_metrics['B_1']
        # case3_B2 = D32_P256_metrics['B_2']
        # # np.save('case3_WE', case3_WE)
        # # np.save('case3_W1', case3_W1)
        # # np.save('case3_W2', case3_W2)
        # # np.save('case3_B1', case3_B1)
        # # np.save('case3_B2', case3_B2)
        # #
        # #
        # # # case3_WE = np.load('case2_WE')
        # # # case3_W1 = np.load('case2_W1')
        # # # case3_W2 = np.load('case2_W2')
        # # # case3_B1 = np.load('case2_B1')
        # # # case3_B2 = np.load('case2_B2')
        # #
        # # # Train Loss
        # # plt.figure()
        # # plt.plot(D32_P256_metrics['TrainLoss'])
        # # plt.xlabel("Epoch")
        # # plt.ylabel("Loss")
        # # plt.title('Training Loss')
        # # plt.legend(['D,P = 32,256'])
        # # plt.savefig("Q2:C:Train Loss.png")
        # # plt.show()
        # # # Train Acc
        # # plt.figure()
        # # plt.plot(D32_P256_metrics['TrainAcc'])
        # # plt.xlabel("Epoch")
        # # plt.ylabel("Loss")
        # # plt.title('Training Acc')
        # # plt.legend(['D,P = 32,256'])
        # # plt.savefig("Q2:C:Train ACc.png")
        # # plt.show()
        # # # Test Loss
        # # plt.figure()
        # # plt.plot(D32_P256_metrics['TestLoss'])
        # # plt.xlabel("Epoch")
        # # plt.ylabel("Loss")
        # # plt.title('Test Loss')
        # # plt.legend(['D,P = 32,256'])
        # plt.savefig("Q2:C:Test Loss.png")
        # plt.show()
        # # Test Acc
        # plt.figure()
        # plt.plot(D32_P256_metrics['TestAcc'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Test Acc')
        # plt.legend(['D,P = 32,256'])
        # plt.savefig("Q2:C:Test Acc.png")
        # plt.show()
        # # Val Loss
        # plt.figure()
        # plt.plot(D32_P256_metrics['ValLoss'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Val Loss')
        # plt.legend(['D,P = 32,256'])
        # plt.savefig("Q2:C:Val Loss.png")
        # plt.show()
        # # Test Acc
        # plt.figure()
        # plt.plot(D32_P256_metrics['ValAcc'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title('Val Acc')
        # plt.legend(['D,P = 32,256'])
        # plt.savefig("Q2:C:Val Acc.png")
        # plt.show()


        print('Question:2:Part:B')

        # Choosing 5 random trigrams from test data
        tri_1 = encoded_test_x[25][:]
        tri_2 = encoded_test_x[91][:]
        tri_3 = encoded_test_x[4242][:]
        tri_4 = encoded_test_x[2424][:]
        tri_5 = encoded_test_x[1746][:]
        case1_WE = np.load('case1_WE.npy')
        case1_W1 = np.load('case1_W1.npy')
        case1_W2 = np.load('case1_W2.npy')
        case1_B1 = np.load('case1_B1.npy')
        case1_B2 = np.load('case1_B2.npy')

        # Obtaining the top 10 predictions with trained weights and biases from case 1
        g1 = top_10_predictions(tri_1, words, case1_WE, case1_W1, case1_W2, case1_B1, case1_B2)
        print('For tri1 ' + str(g1))
        g2 = top_10_predictions(tri_2, words, case1_WE, case1_W1, case1_W2, case1_B1, case1_B2)
        print('For tri2 ' + str(g2))
        g3 = top_10_predictions(tri_3, words, case1_WE, case1_W1, case1_W2, case1_B1, case1_B2)
        print('For tri3 ' + str(g3))
        g4 = top_10_predictions(tri_4, words, case1_WE, case1_W1, case1_W2, case1_B1, case1_B2)
        print('For tri4 ' + str(g4))

        g5 = top_10_predictions(tri_5, words, case1_WE, case1_W1, case1_W2, case1_B1, case1_B2)
        print('For tri5 ' + str(g5))























# Functions used in order
# Hyperbolic tangent activation function
# Functions used in order
# Hyperbolic tangent activation function
def tanh(arr):
    output = (np.exp(arr) - np.exp(-1*arr)) / (np.exp(arr) + np.exp(-1*arr))
    return output
# Derivative of the tanh function
def tanh_derivative(active_out):
    derivative = 1-(active_out**2)
    return derivative
# Classifying the output of the network as 1 and -1
def classify_prediction(y_hat):
    for j in range(len(y_hat)):
        if y_hat[j][0] >= 0:
            y_hat[j][0] = 1
        else:
            y_hat[j][0] = -1
    classified = y_hat
    return classified

# A general Function to implement error metrics for all of the networks in question 1
def error_metrics(image,w1,b1,w2,b2,actual,size,layer,w3,b3):
    if layer == 2:
        test_scores = {'MSE_val': 'nan', 'MCE_val': 'nan'}
        lin_1 = np.dot(image, w1) + b1
        act_1 = tanh(lin_1)
        lin_2 = np.dot(act_1, w2) + b2
        act_2 = tanh(lin_2)
        sq_err = np.square(act_2 - actual)
        sum_sq = np.sum(sq_err)
        mse_value = sum_sq / size
        test_scores['MSE_val'] = mse_value
        pred = classify_prediction(act_2)
        correct = 0
        for check in range(len(actual)):
            if pred[check][0] == actual[check][0]:
                correct = correct + 1
        mce_value = (correct * 100) / size
        test_scores['MCE_val'] = mce_value
        return test_scores
    else:
        test_scores = {'MSE_val': 'nan', 'MCE_val': 'nan'}
        lin_1 = np.dot(image, w1) + b1
        act_1 = tanh(lin_1)
        lin_2 = np.dot(act_1, w2) + b2
        act_2 = tanh(lin_2)
        lin_3 = np.dot(act_2,w3) + b3
        act_3 = tanh(lin_3)
        sq_err = np.square(act_3 - actual)
        sum_sq = np.sum(sq_err)
        mse_value = sum_sq / size
        test_scores['MSE_val'] = mse_value
        pred = classify_prediction(act_3)
        correct = 0
        for check in range(len(actual)):
            if pred[check][0] == actual[check][0]:
                correct = correct + 1
        mce_value = (correct * 100) / size
        test_scores['MCE_val'] = mce_value
        return test_scores

# High/Low Hidden Neuron Size Implementation Function
def neuron_size(neuron,train_ims, train_labs, test_ims,test_labs,epoch,batch_num):
    neuron_metrics = {'MSE_rain': ' ', 'MCE_rain': ' ', 'MSE_tes': ' ', 'MCE_tes': ' '}
    # Initializing network parameters
    h_neuron = neuron
    batch_size = batch_num
    # Using Xavier Initialization for weights and biases std = sqrt(2/input+output)
    std_1 = np.sqrt(2 / (batch_size + h_neuron))
    std_2 = np.sqrt(2 / (1 + h_neuron))
    weights_1 = np.random.normal(0, std_1, size=(1024, h_neuron))
    weights_2 = np.random.normal(0, std_2, size=(h_neuron, 1))
    bias_1 = np.zeros(h_neuron).reshape(1, h_neuron)
    bias_2 = np.zeros(1).reshape(1, 1)
    epochs = epoch
    learning_rate = 0.1
    mce_train_neuron = list()
    mse_train_neuron = list()
    mse_test_neuron = list()
    mce_test_neuron = list()

    #Training

    for test in range(epochs):
        tot = int(math.floor(1900 / batch_size))
        deck = np.random.permutation(len(train_ims))
        shuffled_x = train_ims[deck]
        shuffled_y = train_labs[deck]
        for i in range(tot):
            data_batch = shuffled_x[i * batch_size:i * batch_size + batch_size]
            label_batch = shuffled_y[i * batch_size:i * batch_size + batch_size]

            linear_action_pot_1 = np.dot(data_batch, weights_1) + bias_1
            active_1 = tanh(linear_action_pot_1)

            linear_action_pot_2 = np.dot(active_1, weights_2) + bias_2
            active_2 = tanh(linear_action_pot_2)

            prediction = active_2

            cost = np.square(prediction - label_batch)
            error = prediction - label_batch

            # Backpropagation
            # Output Layer
            # Weight_2 Gradient for output layer dC/dW2 = dL2/dW2 * dA2/dL2 * dC/dA2
            dL2_W2 = active_1
            dA2_L2 = tanh_derivative(active_2)
            dC_A2 = error
            # dZ2 = dC_A2 * dA2_L2
            dZ2 = dA2_L2 * dC_A2
            # Gradient of W2
            grad_W2 = np.dot(dL2_W2.T, dZ2)
            # Bias_2 Gradient dC/dB2 = dL2/dB2 * dA2/dL2 * dC/dA2
            # dL2_B2 = 1
            # Gradient of B2
            grad_B2 = np.sum(dZ2, axis=0, keepdims=True)
            # Updating the weights and biases for Output Layer
            weights_2 -= learning_rate / batch_size * grad_W2
            bias_2 -= learning_rate / batch_size * grad_B2

            # Layer 1
            # Weight_1 Gradient for Hidden layer dC/dW1 = dL2/dW1 * dA2/dL2 * dC/dA2
            # dL2/dW1 = dL1/dW1 * dA1/dL1 * dL2/dA1
            dL1_W1 = data_batch
            dA1_L1 = tanh_derivative(active_1)
            dL2_A1 = weights_2
            dZ1 = np.dot(dZ2, dL2_A1.T) * dA1_L1
            # Gradient of W1
            grad_W1 = np.dot(dL1_W1.T, dZ1)
            # Bias_2 Gradient dC/dB2 = dL2/dB2 * dA2/dL2 * dC/dA2
            dL2_B1 = 1
            # Gradient of B1
            grad_B1 = np.sum(dZ1, axis=0, keepdims=True)
            # Updating the weights and biases for Output Layer
            weights_1 -= learning_rate / batch_size * grad_W1
            bias_1 -= learning_rate / batch_size * grad_B1

        train_score = error_metrics(train_ims, weights_1, bias_1, weights_2, bias_2, train_labs, 1900, 2,
                                    'nan', 'nan')
        mse_training = train_score['MSE_val']
        mce_training = train_score['MCE_val']
        mse_train_neuron.append(mse_training)
        mce_train_neuron.append(mce_training)
        test_score = error_metrics(test_ims, weights_1, bias_1, weights_2, bias_2, test_labs, 1000, 2, 'nan',
                                   'nan')
        mse_testing = test_score['MSE_val']
        mce_testing = test_score['MCE_val']
        mse_test_neuron.append(mse_testing)
        mce_test_neuron.append(mce_testing)

    neuron_metrics['MSE_rain'] = mse_train_neuron
    neuron_metrics['MCE_rain'] = mce_train_neuron
    neuron_metrics['MSE_tes'] = mse_test_neuron
    neuron_metrics['MCE_tes'] = mce_test_neuron

    return neuron_metrics

# Generating a 3 layer network and obtaining its error metrics
def three_layer_network(image, label, weight1, bias1, weight2,bias2,weight3,bias3,batch,epoch,lrn,hid_units,test_im,test_lab,MOM):
    three_layer_metrics = {'train_MSE': 1, 'train_MCE': 1, 'test_MSE': 1, 'test_MCE': 1}
    train_sq = list()
    train_cls = list()
    test_sq = list()
    test_cls = list()

    for ep in range(epoch):
        W3_prev, B3_prev, W2_prev, B2_prev, W1_prev, B1_prev = 0, 0, 0, 0, 0, 0
        tot = int(math.floor(1900/batch))
        shuff = np.random.permutation(len(image))
        rand_x = image[shuff]
        rand_y = label[shuff]
        for i in range(tot):
            image_batch = rand_x[i * batch:i * batch + batch]
            labels_batch = rand_y[i * batch:i * batch + batch]

            lap_1 = np.dot(image_batch, weight1) + bias1
            act_1 = tanh(lap_1)
            lap_2 = np.dot(act_1, weight2) + bias2
            act_2 = tanh(lap_2)
            lap_3 = np.dot(act_2, weight3) + bias3
            act_3 = tanh(lap_3)


            predict = act_3

            error_grad = predict - labels_batch

            # Backpropagation
            # Output Layer
            # Weight_3 Gradient for output layer dC/dW3 = dL3/dW3 * dA3/dL3 * dC/dA3
            dL3_W3 = act_2
            dA3_L3 = tanh_derivative(act_3)
            dC_A3 = error_grad
            # dZ3 = d3_A3 * dA3_L3
            dZ3 = dA3_L3 * dC_A3
            # Gradient of W3
            grad_W3 = np.dot(dL3_W3.T, dZ3)
            # Bias_2 Gradient dC/dB3 = dL2/dB3 * dA3/dL3 * dC/dA3
            # dL3_B3 = 1
            # Gradient of B3
            grad_B3 = np.sum(dZ3, axis=0, keepdims=True)
            # Updating the weights and biases for Output Layer
            # Momentum Terms
            W3_update = (lrn / batch * grad_W3) + (MOM*W3_prev)
            W3_prev = W3_update
            weight3 -= W3_update
            B3_update = (lrn / batch * grad_B3) + (MOM*B3_prev)
            B3_prev = B3_update
            bias3 -= B3_update



            # Layer 2
            # Weight_2 Gradient for Hidden layer dC/dW2 = dL3/dW2 * dA3/dL2 * dC/dA2
            # dL2/dW1 = dL1/dW1 * dA1/dL1 * dL2/dA1
            grad_lap2_to_weight2 = act_1
            grad_act2_to_lap2 = tanh_derivative(act_2)
            grad_lap3_to_act2 = weight3
            dZ_2 = np.dot(dZ3, grad_lap3_to_act2.T) * grad_act2_to_lap2
            # Gradient of W1
            grad_w2 = np.dot(grad_lap2_to_weight2.T, dZ_2)
            # Bias_2 Gradient dC/dB2 = dL2/dB2 * dA2/dL2 * dC/dA2
            # Gradient of B2
            grad_b2 = np.sum(dZ_2, axis=0, keepdims=True)
            # Updating the weights and biases for Hidden Layer2
            W2_update = (lrn / batch * grad_w2) + (MOM*W2_prev)
            W2_prev = W2_update
            weight2 -= W2_update
            B2_update = (lrn / batch * grad_b2) + (MOM*B2_prev)
            B2_prev = B2_update
            bias2 -= B2_update

            # # Layer 1
            lap1_to_w1 = image_batch
            act1_to_lap1 = tanh_derivative(act_1)
            lap2_to_act1 = weight2
            dZ_1 = np.dot(dZ_2, lap2_to_act1.T) * act1_to_lap1
            grad_w1 = np.dot(lap1_to_w1.T, dZ_1)
            grad_b1 = np.sum(dZ_1, axis=0,keepdims=True)
            # Updating the weights and biases
            W1_update = (lrn / batch * grad_w1) + (MOM*W1_prev)
            W1_prev = W1_update
            weight1 -= W1_update
            B1_update = (lrn / batch * grad_b1) + (MOM*B1_prev)
            B1_prev = B1_update
            bias1 -= B1_update

        score_train = error_metrics(image,weight1,bias1,weight2,bias2,label,1900,3,weight3,bias3)
        train_sq.append(score_train['MSE_val'])
        train_cls.append(score_train['MCE_val'])

        score_test = error_metrics(test_im,weight1,bias1,weight2,bias2,test_lab,1000,3,weight3,bias3)
        test_sq.append(score_test['MSE_val'])
        test_cls.append(score_test['MCE_val'])

    # def error_metrics(image, w1, b1, w2, b2, actual, size, layer, w3, b3):
    three_layer_metrics['train_MSE'] = train_sq
    three_layer_metrics['train_MCE'] = train_cls
    three_layer_metrics['test_MSE'] = test_sq
    three_layer_metrics['test_MCE'] = test_cls

    return three_layer_metrics



# Functions Used in Q2

# Activation Function of the 1st hidden Layer
def sigmoid_act(arr):

    return 1/(1+ np.exp(-arr))
# Derivative of sigmoid function
def sigmoid_derv(activ_hid):

    return activ_hid*(1-activ_hid)

# One hot encode function for data and the labels
# option 0 will encode data and 1 will encode labels
def one_hot(data,label,option):
    if option == 0:
        sample = data.shape[0]
        encoder = np.zeros((sample,1,250))
        for i in range(sample):
            unit_o = np.zeros(250).reshape(1,250)
            index_1 = data[i][0] - 1
            index_2 = data[i][1] - 1
            index_3 = data[i][2] - 1
            unit_o[0][index_1] = 1
            unit_o[0][index_2] = 1
            unit_o[0][index_3] = 1
            encoder[i][:][:] = unit_o
    else:
        sample = label.shape[0]
        encoder = np.zeros((sample,250))
        for i in range(sample):
            index = label[i][0] - 1
            encoder[i][index] = 1


    return encoder
# Activation Function of the output Layer
def softmax_act(arr):
    e_z = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
    activated = e_z / np.sum(e_z, axis=-1, keepdims=True)
    return activated

#Derivative of the softmax function
def softmax_derv(activ_out):
    grad_soft = activ_out * (1 - activ_out)
    return grad_soft

# Cross entropy Loss function
def cross_entropy(predict,label):
    sample = predict.shape[0]
    prediction = np.clip(predict, 1e-15, 1 - 1e-15)
    cost = np.sum(-label * np.log(prediction) - (1 - label) * np.log(1 - prediction))
    cr_out = cost/sample
    return cr_out

# Derivative of the Loss function
def cross_derv(predict,label):
    predictions = np.clip(predict, 1e-15, 1 - 1e-15)
    der_ce = - (label/predictions) + (1 - label) / (1 - predictions)
    return der_ce

# Classify the predictions based on the highest probability
def predict_out(activ_output):
    row = activ_output.shape[0]
    col = activ_output.shape[1]
    adjust = np.argmax(activ_output,axis=1)
    prediction_array = np.zeros((row,col))
    for i in range(row):
        prediction_array[i][adjust[i]] = 1

    return prediction_array

#Keeping track of the error metrics over epochs in Loss and Accuracy
def error_metrics(data,label,WE,W1,B1,W2,B2):
    error_values = {'Loss': '', 'ACC': ''}
    sample = data.shape[0]
    Z_E = np.dot(data,WE)
    Z_1 = np.dot(Z_E,W1) + B1
    A_1 = sigmoid_act(Z_1)
    Z_2 = np.dot(A_1,W2) + B2
    A_2 = softmax_act(Z_2)
    epoch_loss = cross_entropy(A_2,label)
    error_values['Loss'] = epoch_loss
    adjust_pred = predict_out(A_2)
    accuracy = cross_check(adjust_pred,label)


    error_values['ACC'] = accuracy
    return error_values
# Stop the algorithm based on cross entropy loss on validation
def cross_check(data,label):
    size = data.shape[0]
    max_dat = np.argmax(data,axis=1)
    max_lab = np.argmax(label,axis=1)
    count = 0
    for k in range(size):
        if max_dat[k] == max_lab[k]:
            count = count + 1

    return count*100/size

# Obtaining the loss over epochs.
def cross_validation(data,label,WE,W1,B1,W2,B2):
    Z_E = np.dot(data, WE)
    Z_1 = np.dot(Z_E, W1) + B1
    A_1 = sigmoid_act(Z_1)
    Z_2 = np.dot(A_1, W2) + B2
    A_2 = softmax_act(Z_2)
    epoch_loss = cross_entropy(A_2, label)
    return epoch_loss


# Question 2 Part A Network Algorithm
def Q2_part_A(D_param, P_param, lrn_rate, mom_coeff, epoch_num, train_data, train_label, test_data, test_label,
              val_data, val_label):
    Q2_metrics = {'TrainLoss': '', 'TrainAcc': '', 'TestLoss': '', 'TestAcc': '', 'ValLoss': '', 'ValAcc': '',
                  'W_E': '', 'W_1': '', 'W_2': '', 'B_1': '', 'B_2': ''}
    sample_size = len(train_data)
    vocab = 250
    inp_neuron = 3
    batch_size = 200
    lrn = lrn_rate
    momentum = mom_coeff
    epochs = epoch_num
    # Embed matrix  (250xD) initialization
    D = D_param
    P = P_param
    # Weights of the Embedding Matrix
    E_weights = np.random.normal(0, 0.01, size=(vocab, D))

    # Network W and B Initializations
    weights_1 = np.random.normal(0, 0.01, size=(D, P))
    weights_2 = np.random.normal(0, 0.01, size=(P, vocab))
    bias_1 = np.random.normal(0, 0.01, size=(1, P))
    bias_2 = np.random.normal(0, 0.01, size=(1, vocab))

    # Error Lists
    loss_train = list()
    accuracy_train = list()
    loss_test = list()
    accuracy_test = list()
    loss_valid = list()
    accuracy_valid = list()
    # train_vs_validation = list()
    # validation_error = list()
    # Forward and Backward Propagation
    for ep in range(epochs):
        print(str(ep))
        tot = int(math.floor(sample_size / batch_size))
        deck = np.random.permutation(len(train_data))
        shuff_x = train_data[deck]
        shuff_y = train_label[deck]
        W_E_prev, W2_prev, B2_prev, W1_prev, B1_prev = 0, 0, 0, 0, 0
        for i in range(tot):
            loc_row = i * batch_size
            loc_col = i * batch_size + batch_size
            data_batch = shuff_x[loc_row:loc_col]  # Shape (200,250)
            label_batch = shuff_y[loc_row:loc_col]  # Shape (200,250)

            # Output of the word embed matrix E_W Shape (200,D)
            embed_ZE = np.dot(data_batch, E_weights)  # Shape (200,D)

            # Sending output of the embed matrix mapping to the hidden layer
            lin1_Z1 = np.dot(embed_ZE, weights_1) + bias_1  # Shape (200,P)
            act_hidden = sigmoid_act(lin1_Z1)  # Shape (200,P)
            lin2_Z2 = np.dot(act_hidden, weights_2) + bias_2  # Shape (200,250)
            act_out = softmax_act(lin2_Z2)  # Shape (200,250)
            prediction = act_out
            loss = cross_entropy(prediction, label_batch)

            # Backpropagation
            # Output Layer
            dA2 = prediction - label_batch  # Shape (200,250)
            dZ2 = dA2 * softmax_derv(act_out)  # Shape (200,250)
            dW2 = np.dot(act_hidden.T, dZ2)  # Shape (P,250)
            dB2 = np.sum(dZ2, axis=0, keepdims=True)
            W2_update = (lrn * dW2 / batch_size) + (W2_prev * momentum)
            W2_prev = W2_update
            weights_2 -= W2_update
            B2_update = (lrn * dB2 / batch_size) + (B2_prev * momentum)  # Shape (1,250)
            B2_prev = B2_update
            bias_2 -= B2_update

            # Hidden Layer
            dA1 = np.dot(dZ2, weights_2.T)  # Shape (200,P)
            dZ1 = dA1 * sigmoid_derv(act_hidden)  # Shape (200,P)
            dW1 = np.dot(embed_ZE.T, dZ1)  # Shape (D,P)
            dB1 = np.sum(dZ1, axis=0, keepdims=True)
            W1_update = (lrn * dW1 / batch_size) + (W1_prev * momentum)
            W1_prev = W1_update
            weights_1 -= W1_update
            B1_update = (lrn * dB1 / batch_size) + (B1_prev * momentum)
            B1_prev = B1_update
            bias_1 -= B1_update

            # Embed Layer
            dA_E = np.dot(dZ1, weights_1.T)  # Shape (200,D)
            dZ_E = dA_E  # Shape (200,D)
            dW_E = np.dot(data_batch.T, dZ_E)
            W_E_update = (lrn * dW_E / batch_size) + (momentum * W_E_prev)
            W_E_prev = W_E_update
            E_weights -= W_E_update

        # Training Error Metrics
        accio_train = error_metrics(train_data, train_label, E_weights, weights_1, bias_1, weights_2, bias_2)
        loss_train.append(accio_train['Loss'])
        training_loss = accio_train['Loss']
        accuracy_train.append(accio_train['ACC'])
        print(str(ep) + ' Loss= ' + str(accio_train['Loss']) + ' Accuracy = ' + str(accio_train['ACC']))
        # Testing Error Metrics
        accio_test = error_metrics(test_data, test_label, E_weights, weights_1, bias_1, weights_2, bias_2)
        loss_test.append(accio_test['Loss'])
        accuracy_test.append(accio_test['ACC'])

        # Validation Error Metrics
        accio_valid = error_metrics(val_data, val_label, E_weights, weights_1, bias_1, weights_2, bias_2)
        loss_valid.append(accio_valid['Loss'])
        validation_loss = accio_valid['Loss']
        accuracy_valid.append(accio_valid['ACC'])

        # validation_entropy = cross_validation(encoded_val_x, encoded_val_d, E_weights, weights_1, bias_1, weights_2,
        #                                       bias_2)
        # print('Epoch:' + str(ep) + ': Validation Error = ' + str(validation_entropy))
        # validation_error.append(validation_entropy)
        # Stopping the Algorithm on cross_validation
        print('Epoch:' + str(ep) + ' Validation Loss:' + str(validation_loss))
        # if ep > 10:
        #     if validation_loss > loss_valid[int(ep) - 1]:
        #         print('Algorithm Stopped on Epoch ' + str(ep))
        #         break

    Q2_metrics['TrainLoss'] = loss_train
    Q2_metrics['TrainAcc'] = accuracy_train
    Q2_metrics['TestLoss'] = loss_test
    Q2_metrics['TestAcc'] = accuracy_test
    Q2_metrics['ValLoss'] = loss_valid
    Q2_metrics['ValAcc'] = accuracy_valid
    # Q2_metrics['W_E'] = E_weights
    # Q2_metrics['W_1'] = weights_1
    # Q2_metrics['W_2'] = weights_2
    # Q2_metrics['B_1'] = bias_1
    # Q2_metrics['B_2'] = bias_2

    return Q2_metrics
# Function that generates top 10 predictions and their corresponding words along with input words
def top_10_predictions(test_data,word_data,EW,W1,W2,B1,B2):
    guess_net = {'Word1': '','Word2': '', 'Word3': '', 'Word4': ''}
    test_data_indices = test_data.reshape(250,)
    word_index_in = list()
    prediction_index = list()
    for i in range(250):
        if test_data_indices[i] == 1:
            word_index_in.append(i)

    # guess_net['IN'] = word_index_in
    guess_net['Word1'] = word_data[word_index_in[0]]
    guess_net['Word2'] = word_data[word_index_in[1]]
    guess_net['Word3'] = word_data[word_index_in[2]]
    Z_E = np.dot(test_data,EW)
    Z1 = np.dot(Z_E,W1) + B1
    A1 = sigmoid_act(Z1)
    Z2 = np.dot(A1,W2) + B2
    A2 = softmax_act(Z2)
    word_predict = A2

    # obtaining 10 highest probabilities
    guess_indices = word_predict.reshape(250,)
    top_10 = np.argpartition(guess_indices,-10)[-10:]
    top_10_probabilities = list()
    for k in range(10):
        top_10_probabilities.append(word_data[top_10[k]])
    guess_net['Word4'] = top_10_probabilities
    return guess_net
arman_budunoglu_21602635_hw2(question)