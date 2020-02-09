from model import StyleTransferModel
import torchvision.transforms as transforms
from io import BytesIO
from telegram_token import token

# В бейзлайне пример того, как мы можем обрабатывать две картинки, пришедшие от пользователя.
unloader = transforms.ToPILImage()
model = StyleTransferModel()
first_image_file = {}

CONTENT, STYLE = range(2)


def imsave(tensor, title="out.jpg"):
    image = tensor.cpu().clone()
    image = image.squeeze(0)      # функция для отрисовки изображения
    image = unloader(image)
    image.save(title+".jpg", "JPEG", quality=80, optimize=True, progressive=True)
 


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Пришли фото которое хочешь обработать")
    return CONTENT



def photo_content(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('content_photo.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'content_photo.jpg')
    update.message.reply_text('теперь фото стиля')

    return STYLE

def photo_style(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('style_photo.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'style_photo.jpg')
    update.message.reply_text('перенос стиля скоро будет')
    style_img = model.image_loader("style_photo.jpg")# as well as here
    content_img = model.image_loader('content_photo.jpg')#измените путь на тот который у вас.
    output = model.run_style_transfer(content_img, style_img, content_img)
    imsave(output, "out")
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('out.jpg', 'rb'))
    return CONTENT


def cancel(update, context):
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('Bye! I hope we can talk again some day.')

    return ConversationHandler.END
def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters, ConversationHandler, CommandHandler
    import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)

    

    logger = logging.getLogger(__name__)
    updater = Updater(token=token,  request_kwargs={'proxy_url': 'socks5h://163.172.152.192:1080'}, use_context=True)
    dp = updater.dispatcher
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            CONTENT: [MessageHandler(Filters.photo, photo_content)],

            STYLE: [MessageHandler(Filters.photo,photo_style)],
            
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)

    # log all errors
    dp.add_error_handler(error)

    updater.start_polling()

