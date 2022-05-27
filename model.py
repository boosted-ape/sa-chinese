import os
import io
import numpy
import tensorflow as tf
from tensorflow_text import HubModuleTokenizer
from keras.models import Sequential
from keras import layers

MODEL_HANDLE = "https://tfhub.dev/google/zh_segmentation/1"
BUFFER_SIZE = 50000
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE
segmenter = HubModuleTokenizer(MODEL_HANDLE)

checkpoint_dir = "checkpoint"

latest = tf.train.latest_checkpoint(checkpoint_dir)

def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)

def labeler(example, index):
    return example, tf.cast(index, tf.int64)

punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.《"
file = open("stopword.txt")
stopwords = ''.join(file.read().splitlines())
file.close()
punc = punc + stopwords

example_sentence = "（早报讯）我国对所有完成冠病疫苗接种者重开边境一个月来，樟宜机场接待了更多旅客，4月底的航空乘客量进一步增加至疫前水平的近40％，相比3月的约18％增加了超过一倍。乘客量下来几个月还会继续增长，相信今年内可如预期恢复到疫前水平的至少50%。交通部长易华仁今午（5月4日）在樟宜机场集团的“樟宜航空大奖”颁奖礼上致辞时指出，我国不仅要恢复以前的航空衔接，也要进一步扩大航空网络、增加航班的密集度，以及与航空公司建立新的伙伴关系。我国4月起实行简化的疫苗接种者旅游框架，所有已完成冠病疫苗接种的旅客，不论来自哪个国家或地区都可入境新加坡，4月26日起也不再须要接受行前冠病检测或隔离。"


embedding_weights = numpy.genfromtxt('vectors.tsv', delimiter="\t")
vocab_size = embedding_weights.shape[0] + 1

os.listdir(checkpoint_dir)

preprocess_layer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    split=segmenter.tokenize,
    output_mode='int',
    output_sequence_length=250)

preprocess_layer.set_vocabulary("metadata.tsv")




test_dataset_0 = tf.data.TextLineDataset(
    "/home/bj/sentiment-analysis/chnsenticorp/test/0/0.txt")
test_dataset_1 = tf.data.TextLineDataset(
    "/home/bj/sentiment-analysis/chnsenticorp/test/1/1.txt")

test_dataset_0 = test_dataset_0.map(
    lambda x: tf.strings.regex_replace(x, "[%s]+" % punc, " "))
test_dataset_1 = test_dataset_1.map(
    lambda x: tf.strings.regex_replace(x, "[%s]+" % punc, " "))

test_dataset_0 = test_dataset_0.map(lambda x: labeler(x, 0))
test_dataset_1 = test_dataset_1.map(lambda x: labeler(x, 1))

test_labeled_data_sets = test_dataset_0.concatenate(test_dataset_1)
test_labeled_data_sets = test_labeled_data_sets.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

test_ds = test_labeled_data_sets.take(832).batch(BATCH_SIZE)
test_ds = configure_dataset(test_ds)

john= preprocess_layer.get_vocabulary()

new_model = Sequential([preprocess_layer,tf.keras.models.load_model("save_model/my_model")])
new_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
loss, accuracy = new_model.evaluate(test_ds)

new_model.summary()


print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))


loss, accuracy = new_model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))