#================================
from textgenrnn import textgenrnn
from datetime import datetime
#================================
model_name = 'Take_1'

textgen = textgenrnn(weights_path='Take_1_weights.hdf5',
                       vocab_path='Take_1_vocab.json',
                       config_path='Take_1_config.json')
n = 1
max_gen_length = 500 
 
temperature = [1.0,0.5,0.2,0.2] 
prefix = None   # if you want each generated text to start with a given seed text
timestring = datetime.now().strftime('%Y%m%d_%H%M%S')   
gen_file = '{}_gentext_{}.txt'.format(model_name, timestring)
textgen.generate_to_file(gen_file,
                         temperature=temperature,
                         prefix=prefix,
                         n=n,
                         max_gen_length=max_gen_length)

