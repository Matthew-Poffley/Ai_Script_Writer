#================================
def training(train_cfg, model_cfg, model_name, file_name, batch):
    from textgenrnn import textgenrnn
    textgen = textgenrnn(name= model_name)

    train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file

    train_function(
        file_path=file_name,
        new_model=True,
        num_epochs=train_cfg['num_epochs'],
        gen_epochs=train_cfg['gen_epochs'],
        batch_size= batch,
        train_size=train_cfg['train_size'],
        dropout=train_cfg['dropout'],
        validation=train_cfg['validation'],
        is_csv=train_cfg['is_csv'],
        rnn_layers=model_cfg['rnn_layers'],
        rnn_size=model_cfg['rnn_size'],
        rnn_bidirectional=model_cfg['rnn_bidirectional'],
        max_length=model_cfg['max_length'],
        dim_embeddings=100,
        word_level=model_cfg['word_level'])
    return textgen
#================================