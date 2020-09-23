#================================
def typey(temp, train_cfg, model_cfg, model_name, prefix,textgen):
    from datetime import datetime

    temperature = temp 
    prefix = None   # if you want each generated text to start with a given seed text
    
    if train_cfg['line_delimited']:
        n = 1000
        max_gen_length = 60 if model_cfg['word_level'] else 300
    else:
            n = 1
            max_gen_length = 2000 if model_cfg['word_level'] else 10000
            
    timestring = datetime.now().strftime('%Y%m%d_%H%M%S')
    gen_file = '{}_gentext_{}.txt'.format(model_name, timestring)
            
    textgen.generate_to_file(gen_file,
                         temperature=temperature,
                         prefix=prefix,
                         n=n,
                         max_gen_length=max_gen_length)
#================================