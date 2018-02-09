import os
import argparse
import pickle
import numpy as np

import models
from data import load_data


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your models.")

    parser.add_argument("--data", type=str, required=True, help="The data file to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create (for training) or load (for testing).")
    parser.add_argument("--algorithm", type=str,
                        help="The name of the algorithm to use. (Only used for training; inferred from the model file at test time.)")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create. (Only used for testing.)")

    
    parser.add_argument("--online-learning-rate", type=float, help="The learning rate", default=1)
    parser.add_argument("--online-training-iterations", type=int, help="The number of training iterations for online methods.", default=5)
    args = parser.parse_args()

    return args


def check_args(args):
    mandatory_args = {'data', 'mode', 'model_file', 'algorithm', 'predictions_file'}
    if not mandatory_args.issubset(set(dir(args))):
        raise Exception('Arguments that we provided are now renamed or missing. If you hand this in, you will get 0 points.')

    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--model should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--predictions-file should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def main():
    args = get_args()
    check_args(args)

    if args.mode.lower() == "train":
        # Load the training data.
        X, y = load_data(args.data)

        # Create the model.
        # TODO: Add other algorithms as necessary.
        if args.algorithm.lower() == 'useless':
            model = models.Useless()
        elif args.algorithm.lower() == 'majority':
            model = models.Majority()
        elif args.algorithm.lower() == 'perceptron':     
            model = models.Perceptron(args.online_learning_rate, args.online_training_iterations)
        elif args.algorithm.lower() == 'lr_sgd':
            optimizer = "sgd"
            model = models.Logistic(optimizer, args.online_learning_rate, args.online_training_iterations)
        elif args.algorithm.lower() == 'lr_adam':
            optimizer ="adam"
            model = models.Logistic(optimizer, args.online_learning_rate, args.online_training_iterations)
        else:
            raise Exception('The model given by --model is not yet supported.')

        # Train the model.
        model.fit(X, y)

        # Save the model.
        try:
            with open(args.model_file, 'wb') as f:
                pickle.dump(model, f)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping model pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        X, y = load_data(args.data)

        # Load the model.
        try:
            with open(args.model_file, 'rb') as f:
                model = pickle.load(f)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading model pickle.")

        # Compute and save the predictions.
        y_hat = model.predict(X)
        invalid_label_mask = (y_hat != 0) & (y_hat != 1)
        if any(invalid_label_mask):
            raise Exception('All predictions must be 0 or 1, but found other predictions.')
        np.savetxt(args.predictions_file, y_hat, fmt='%d')
            
    else:
        raise Exception("Mode given by --mode is unrecognized.")


if __name__ == "__main__":
    main()
