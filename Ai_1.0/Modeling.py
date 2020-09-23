#================================
def model1(rnn_size, rnn_layers, max_length):
                model_cfg = {
                                'word_level': False,   # set to True if want to train a word-level model (requires more data and smaller max_length)
                                'rnn_size': rnn_size,   # number of LSTM cells of each layer (128/256 recommended)
                                'rnn_layers': rnn_layers,   # number of LSTM layers (>=2 recommended)
                                'rnn_bidirectional': False,   # consider text both forwards and backward, can give a training boost
                                'max_length': max_length,   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
                                'max_words': 100000,   # maximum number of words to model; the rest will be ignored (word-level model only)
                                }
                return model_cfg
#================================
def train1(epochs, train_size, dropout):
        train_cfg = {
                        'line_delimited': False,   # set to True if each text has its own line in the source file
                        'num_epochs': epochs,   # set higher to train the model for longer
                        'gen_epochs': epochs,   # generates sample text from model after given number of epochs
                        'train_size': train_size,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
                        'dropout': dropout,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
                        'validation': False,   # If train__size < 1.0, test on holdout dataset; will make overall training slower
                        'is_csv': False   # set to True if file is a CSV exported from Excel/BigQuery/pandas
                    }
        return train_cfg
#================================

    
