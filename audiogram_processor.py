import pandas as pd
import csv
import numpy as np
import random
from typing import Tuple
import tensorflow as tf
import tensorflow_addons as tfa
import sklearn.model_selection
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import itertools


'''
    The Data class is used to ingest the data in the CSV file and clean and split the data so it is
    easy to use in the machine learning pipeline.
'''
class Data:

    def __init__(self, path: str):
        self._path = None
        self._data = None

        self._train_inputs = None
        self._train_targets = None
        
        self._test_inputs = None
        self._test_targets = None
        
        self._val_inputs = None
        self._val_targets = None
        
        self.process_data(path)

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    @property
    def train_inputs(self) -> np.ndarray:
        return self._train_inputs.copy()

    @property
    def train_outputs(self) -> np.ndarray:
        return self._train_outputs.copy()

    @property
    def test_inputs(self) -> np.ndarray:
        return self._test_inputs.copy()

    @property
    def test_outputs(self) -> np.ndarray:
        return self._test_outputs.copy()

    @property
    def val_inputs(self) -> np.ndarray:
        return self._val_inputs.copy()

    @property
    def val_outputs(self) -> np.ndarray:
        return self._val_outputs.copy()

    # This is the main function that loads, cleans, and divides the data
    def process_data(self, path: str):
        
        self.load_data(path)
        self.clean_data()
        self.divide_data()

    # Load the data from the CSV into a dataframe where we can process the data
    def load_data(self, path: str, nrows = None):
        
        self._path = path
        if nrows:
            self._data = pd.read_csv(path, nrows = nrows)
        else:
            self._data = pd.read_csv(path)

    # Cleans the data and replace non-integer values with NaNs
    # This operation covers a couple of cases. In the data, there are two bad cases:
        # 1. '**' for null inputs
        # 2. Somewhere in the data there are strings
    
    def clean_data(self):
    
        freqs = ['L500k', 'L1k', 'L2k', 'L3k', 'L4k', 'L6k', 'L8k', 'R500k', 'R1k', 'R2k', 'R3k', 'R4k', 'R6k', 'R8k']
        self._data = self._data[freqs]

        
        self._data = self._data.apply(lambda x:pd.to_numeric(x, errors = 'coerce'))

        # Then we can drop all rows with NaNs
        self._data.dropna(inplace = True)
        self._data.reset_index(drop = False, inplace = True)

    # Divide the data into inputs and outputs. The inputs are measured frequencies, and outputs are all other
    # frequencies. We will treat left and right data as separate measurements / outputs. Then shuffle the data
    # and split it into train/test/validation sets for use in training and testing.
    def divide_data(self):

        # Get the input columns of the data
        left_inputs = self._data[['L2k', 'L4k', 'L6k']].to_numpy()
        right_inputs = self._data[['R2k', 'R4k', 'R6k']].to_numpy()
        inputs = np.concatenate((left_inputs, right_inputs), axis = 0)

        # Get the output columns of the data
        left_outputs = self._data[['L500k', 'L1k', 'L3k', 'L8k']].to_numpy()
        right_outputs = self._data[['R500k', 'R1k', 'R3k', 'R8k']].to_numpy()
        outputs = np.concatenate((left_outputs, right_outputs), axis = 0)

        # Convert the decibel values, which are multiples of 5, to integers in the range [0, ..., n_discrete_volumes].
        # This way we can make it a classification problem.
        matrix_min = np.min(np.min(outputs))
        outputs = ((outputs - matrix_min)/5).astype('int')

        # Shuffle the indices for selecting the individual sets (train/test/validate)
        n = len(inputs)
        indices = list(range(n))
        random.shuffle(indices)

        # Get the number of data points for each set
        n_train = int(.75 * n)
        n_test = int(.15 * n)
        n_val = n - (n_train + n_test)

        # Get the indices for each set
        msk_train = indices[0:n_train]
        msk_test = indices[n_train:(n_train + n_test)]
        msk_val = indices[(n_train + n_test):]

        # Select the data points for each set
        self._train_inputs = inputs[msk_train, :]
        self._train_outputs = outputs[msk_train, :]

        self._test_inputs = inputs[msk_test, :]
        self._test_outputs = outputs[msk_test, :]

        self._val_inputs = inputs[msk_val, :]
        self._val_outputs = outputs[msk_val, :]

    
        def get_dict(input_array):
            output_dict =  {}
            output_dict['500k'] = input_array[:, 0]
            output_dict['1k'] = input_array[:, 1]
            output_dict['3k'] = input_array[:, 2]
            output_dict['8k'] = input_array[:, 3]

            return output_dict

        self._train_outputs = get_dict(self._train_outputs)
        self._test_outputs = get_dict(self._test_outputs)
        self._val_outputs = get_dict(self._val_outputs)



'''
    The Model class will hold 4 models, each parameterized to predict the unknown threshold volumes for 
    500k, 1k, 3k, and 8k, respectively.

    Each model will accept as input the measured volumes for the 2k, 4k, and 6k frequencies and predict
    a threshold volume for its designated frequency.
'''
class Model:

    def __init__(self, n_samples):
        self.input_dim = n_samples
        self.models =  {'500k':self.classification_model(), '1k':self.classification_model(), 
                        '3k':self.classification_model(), '8k':self.classification_model()}

        self.accuracies =  {}
        self.callbacks = self.create_call_backs()

    # Create callbacks for tensorboard
    def create_call_backs(self):
        callbacks = [TensorBoard(log_dir = './logs/', histogram_freq = 1, 
        write_graph = True, write_images = True, 
        update_freq = 'epoch', profile_batch = 2)]

        return callbacks
    
    # Create the one-hot vector
    def get_one_hot_vector(self, output_train):
        
        hotvector = np.zeros((len(output_train), 26))
        for i, value in enumerate(output_train):
            hotvector[i][value] = 1

        return hotvector
    
    # Create the classification model that we will use to predict volumes for each output frequency
    def classification_model(self):
        
        model = tf.keras.Sequential(name=str(random.randint(10000, 99999)))
        model.add(tf.keras.layers.Dense(156, input_dim = self.input_dim, activation = 'relu'))
        model.add(tf.keras.layers.Dense(78, input_dim = self.input_dim, activation = 'relu'))
        model.add(tf.keras.layers.Dense(26, activation = 'softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        model.summary()
        return model

    # Train a model for a specific frequency
    def run_model(self, input_train, output_train, freq):
        
        hotvector_output_train = self.get_one_hot_vector(output_train)
        self.models[freq].fit(input_train, hotvector_output_train, verbose = 1, epochs = 150, callbacks = self.callbacks)

    # Evaluate the training of the model using the validation data
    def evaluate_model(self, input_val, output_val, freq):
        
        hotvector_output_val = self.get_one_hot_vector(output_val)
        val_loss, val_accuracy = self.models[freq].evaluate(input_val, hotvector_output_val, verbose = 2, callbacks = self.callbacks)
        
        print('Validation Accuracy: %.2f' % (val_accuracy * 100))


    # Plots the confusion matrix for the data
    def plot_confusion_matrix(cm, target_names, title = 'Confusion matrix', 
        cmap = None, normalize = False):

        accuracy = np.trace(cm)/float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Greens')

        plt.figure(figsize = (8, 6))
        plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation = 45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis]

        thresh = cm.max()/1.5 if normalize else cm.max()/2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]), 
                horizontalalignment = "center", 
                color = "white"if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]), 
                horizontalalignment = "center", 
                color = "white"if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()



    def test_model(self, input_test, output_test, freq):
        # Test the model using the test data
        predicted_outputs = self.models[freq].predict(input_test, callbacks = self.callbacks)

        predictions = np.argmax(predicted_outputs, axis = 1)

        a = sklearn.metrics.accuracy_score(predictions, output_test)
        print('accuracy: ', a * 100)

        return a

    
    # Train the four models, one for each output frequency
    def run_model_for_all_freqs(self, input_train, input_test, input_val, output_train, output_test, output_val):
        
        for freq in output_train.keys():
            print('Training the model and predicting for ', freq)

            self.run_model(input_train, output_train[freq], freq)
            self.evaluate_model(input_val, output_val[freq], freq)
            acc = self.test_model(input_test, output_test[freq], freq)
            self.accuracies[freq] = acc

        print(self.accuracies)


    # Predict the volumes for the four missing frequencies using the three measured frequencies
    def predict_volumes(self, vol2k, vol4k, vol6k):
        
        out_volumes =  {}
        for out_freq, model in self.models.items():
            vold_ind = np.argmax(model.predict(np.array([[vol2k, vol4k, vol6k]])))
            volume = vold_ind * 5

            out_volumes[out_freq] = volume

        return out_volumes


if __name__ == '__main__':
    from tensorboard import program

    tracking_address = './logs'
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    dataset = Data('./Paper3_WebData_Final.csv')
    m = Model(dataset.train_inputs[0].shape[0])
    m.run_model_for_all_freqs(dataset.train_inputs, dataset.test_inputs, dataset.val_inputs, dataset.train_outputs, dataset.test_outputs, dataset.val_outputs)
    
    print('\n')
    print('Input volumes 25, 30, 55 decibels for 2k, 4k, 6k frequencies, respectively:')
    print("Predicted: ", m.predict_volumes(25,30,55), " decibels")


