#================================
import Modeling
import Training
import Typer
#================================
file_name = "C:/Users/mrsta/Documents/Python Scripts/FnF.txt"         # Label the .txt files to learn
model_name =  "Take_3"                                                         # Name the model so it can be reused
#================================
rnn_size = 128                                                                 # Set the rnn_size 
rnn_layers = 3                                                                 # Set the depth of the alogrithm 
max_length = 30                                                                # Number of characters/ words it can read to make a guess
epochs = 20                                                                    # Number of epochs
train_size = 0.8                                                              
dropout = 0
batch = 1024                                                                   # Batch size
prefix = None                                                                  # The first word to generate off
temp = [1.0, 0.5,0.2,0.2]                                                      # How "crazy" the guessing will be
#================================
model_cfg = Modeling.model1(rnn_size, rnn_layers, max_length)
train_cfg = Modeling.train1(epochs, train_size, dropout)
textgen = Training.training(train_cfg, model_cfg, model_name, file_name, batch)
Typer.typey(temp, train_cfg, model_cfg, model_name, prefix,textgen)
#================================
