## Data Processing:

The data is first divided into input and output categories where input includes the frequencies 2k, 4k and 6k and the output frequencies include 500k, 1k, 3k and 8k. The left and right data are considered separate audiograms hence are appended to each other. Each group is then shuffled and divided into train (70%), validation (15%) and test (15%). 

## Model

The model consists of three dense layers with the relu activation, softmax for the output layer and categorical cross entropy for the loss function. For each individual output frequency the model is trained on train and validation data. The model then returns the test accuracy for that frequency. All the output frequency accuracies and model are saved in class variables and can be used by the predict_volumes() method to make predictions for a given input data. 

## How to Run

Pre-requisite: Python 3.10

1. Include the data file in the same directory as audiogram_processor.py `
2. Run the following command:
` $ python audiogram_processor.py `

### Visualization using Tensorboard:


