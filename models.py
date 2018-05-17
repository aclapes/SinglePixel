from keras.layers import Bidirectional, Masking, LSTM, GRU, Dense, Activation
from keras import Sequential

def init_model(lstm_variation, input_shape, hidden_size, nr_classes, dropout=0.2, recurrent_dropout=0.2):
    # Model definition
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    # (lstm)
    if lstm_variation == 'onelayer_lstm':
        model.add(LSTM(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout))
    elif lstm_variation == 'twolayer_lstm':
        model.add(LSTM(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
        model.add(LSTM(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout))
    elif lstm_variation == 'onelayer_bilstm':

        model.add(Bidirectional(LSTM(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    elif lstm_variation == 'twolayer_bilstm':
        model.add(Bidirectional(LSTM(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
        model.add(Bidirectional(LSTM(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    # (gru)
    elif lstm_variation == 'onelayer_gru':
        model.add(GRU(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout))
    elif lstm_variation == 'twolayer_gru':
        model.add(GRU(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
        model.add(GRU(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout))
    elif lstm_variation == 'onelayer_bigru':
        model.add(Bidirectional(GRU(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    elif lstm_variation == 'twolayer_bigru':
        model.add(Bidirectional(GRU(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
        model.add(Bidirectional(GRU(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    else:
        raise NotImplementedError

    model.add(Dense(nr_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def scheduler(iteration):
    if iteration < 10:
        return 1e-3
    elif iteration < 20:
        return 5e-4
    else:
        return 1e-4