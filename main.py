from keras import layers
from keras.models import Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from matplotlib import collections
from tensorflow_text import HubModuleTokenizer
import tensorflow as tf
import collections
import numpy


example_sentence = "（早报讯）我国对所有完成冠病疫苗接种者重开边境一个月来，樟宜机场接待了更多旅客，4月底的航空乘客量进一步增加至疫前水平的近40％，相比3月的约18％增加了超过一倍。乘客量下来几个月还会继续增长，相信今年内可如预期恢复到疫前水平的至少50%。交通部长易华仁今午（5月4日）在樟宜机场集团的“樟宜航空大奖”颁奖礼上致辞时指出，我国不仅要恢复以前的航空衔接，也要进一步扩大航空网络、增加航班的密集度，以及与航空公司建立新的伙伴关系。我国4月起实行简化的疫苗接种者旅游框架，所有已完成冠病疫苗接种的旅客，不论来自哪个国家或地区都可入境新加坡，4月26日起也不再须要接受行前冠病检测或隔离。"
VOCAB_SIZE = 50000
max_len = 200

BUFFER_SIZE = 50000
BATCH_SIZE = 64
VALIDATION_SIZE = 5000
AUTOTUNE = tf.data.AUTOTUNE


train_dataset_0 = tf.data.TextLineDataset(
    "/home/bj/sentiment-analysis/chnsenticorp/train/0/0.txt")
train_dataset_1 = tf.data.TextLineDataset(
    "/home/bj/sentiment-analysis/chnsenticorp/train/1/1.txt")

MODEL_HANDLE = "https://tfhub.dev/google/zh_segmentation/1"
segmenter = HubModuleTokenizer(MODEL_HANDLE)


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)


punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."



train_dataset_0 = train_dataset_0.map(lambda x: tf.strings.regex_replace(x, "[%s]+" %punc, " "))
train_dataset_1 = train_dataset_1.map(lambda x: tf.strings.regex_replace(x, "[%s]+" %punc, " "))

train_dataset_0 = train_dataset_0.map(lambda x: segmenter.tokenize(x))
train_dataset_1 = train_dataset_1.map(lambda x: segmenter.tokenize(x))

train_dataset_0 = train_dataset_0.map(lambda x: labeler(x, 0))
train_dataset_1 = train_dataset_1.map(lambda x: labeler(x, 1))

labeled_data_sets = [train_dataset_0, train_dataset_1]

tokenised_ds = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    tokenised_ds = tokenised_ds.concatenate(labeled_dataset)

tokenised_ds = tokenised_ds.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

tokenised_ds = configure_dataset(tokenised_ds)
vocab_dict = collections.defaultdict(lambda: 0)
for toks in tokenised_ds.as_numpy_iterator():
    for tok in toks:
        if (type(tok) == numpy.ndarray):
            for word in tok:
                vocab_dict[word] += 1

vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
vocab = [token for token, count in vocab]
vocab = vocab[:VOCAB_SIZE]
vocab_size = len(vocab)

print("Vocab size: ", vocab_size)
for i in range(5):
    print(vocab[i].decode("utf-8"))


model = Sequential()
model.add(layers.Embedding(VOCAB_SIZE, 20))
model.add(layers.LSTM(15, dropout=0.5))
model.add(layers.Dense(3, activation='relu'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("model.hdf5", monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='auto', period=1, save_weights_only=False)

# istory = model.fit(train_dataset, epochs=70, validation_data=(
#   test_dataset), callbacks=[checkpoint])
