
import telebot
import traceback
import config
from handler import *
from keras.models import load_model
from PIL import Image, ImageOps
import tensorflow as tf

bot = telebot.TeleBot(config.TOKEN)
classes = ["съедобный", "несъедобный"]
model = load_model('mushrooms_final.h5')


def get_photo(message):
    photo = message.photo[1].file_id
    file_info = bot.get_file(photo)
    file_content = bot.download_file(file_info.file_path)
    return file_content


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,
                     'Пришли фото боту, а нейронная сеть определит съедобный ли это гриб.')


@bot.message_handler(content_types=['photo'])
def repeat_all_messages(message):
    try:
        file_content = get_photo(message)
        image = byte2image(file_content)

        size = (180, 180)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict_on_batch(img_array).flatten()
        predictions = tf.nn.sigmoid(predictions)
        predicted_class = tf.argmax(predictions)

        bot.send_message(message.chat.id, text=f'Скорее всего этот гриб {classes[predicted_class]}')

    except Exception:
        traceback.print_exc()
        bot.send_message(message.chat.id, 'Ошибка! Что то пошло не по плану.')


if __name__ == '__main__':
    import time

    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as e:
            time.sleep(15)
            print('Restart!')