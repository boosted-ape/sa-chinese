from tabnanny import check
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.models import Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import jieba.posseg as pseg

string = "（早报讯）我国对所有完成冠病疫苗接种者重开边境一个月来，樟宜机场接待了更多旅客，4月底的航空乘客量进一步增加至疫前水平的近40％，相比3月的约18％增加了超过一倍。乘客量下来几个月还会继续增长，相信今年内可如预期恢复到疫前水平的至少50%。交通部长易华仁今午（5月4日）在樟宜机场集团的“樟宜航空大奖”颁奖礼上致辞时指出，我国不仅要恢复以前的航空衔接，也要进一步扩大航空网络、增加航班的密集度，以及与航空公司建立新的伙伴关系。我国4月起实行简化的疫苗接种者旅游框架，所有已完成冠病疫苗接种的旅客，不论来自哪个国家或地区都可入境新加坡，4月26日起也不再须要接受行前冠病检测或隔离。"
max_words = 5000
max_len = 200


def analyzer(string):
    words = pseg.cut(string)
    word_sequence = []

    for w in words:
        if(w.flag == "x"):
            continue
        word_sequence.append(w.word)

    return word_sequence


tokenizer = Tokenizer(num_words=max_words)
word_sequence = analyzer(string)
tokenizer.fit_on_texts(word_sequence)
sequences = tokenizer.texts_to_sequences(word_sequence)
words = pad_sequences(sequences, maxlen=max_len)

model = Sequential()
model.add(layers.Embedding(max_words, 20))
model.add(layers.LSTM(15, dropout=0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("model.hdf5", monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='auto', period=1, save_weights_only=False)

history = model.fit(X_train, Y_train, epochs=70, validation_data=(
    X_test, Y_test), callbacks=[checkpoint])
