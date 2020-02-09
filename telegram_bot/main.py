from model import StyleTransferModel
import torchvision.transforms as transforms
from io import BytesIO
from telegram_token import token
import os 

from telegram.ext.dispatcher import run_async
# В бейзлайне пример того, как мы можем обрабатывать две картинки, пришедшие от пользователя.
unloader = transforms.ToPILImage()
model = StyleTransferModel()
first_image_file = {}

CONTENT, STYLE = range(2)
PHOTOS_FOLDER = './photos/'

def imsave(tensor, title="out.jpg"):
    image = tensor.cpu().clone()
    image = image.squeeze(0)      # функция для отрисовки изображения
    image = unloader(image)
    image.save(title, "JPEG", quality=80, optimize=True, progressive=True)
 
def imdel(title):
    if os.path.exists(title):
        os.remove(title)

@run_async
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Пришли фото которое хочешь обработать📷")
    return CONTENT


@run_async
def photo_content(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    path_content = os.path.join(PHOTOS_FOLDER, '{}_content_photo.jpg'.format(str(update.message.from_user.username)))
    photo_file.download(path_content)
    logger.info("Photo of %s: %s", user.first_name, 'content_photo.jpg')
    update.message.reply_text('Теперь фото стиля🎆')

    return STYLE

@run_async
def photo_style(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    path_style = os.path.join(PHOTOS_FOLDER, '{}_style_photo.jpg'.format(str(update.message.from_user.username)))
    path_content = os.path.join(PHOTOS_FOLDER, '{}_content_photo.jpg'.format(str(update.message.from_user.username)))
    path_out = os.path.join(PHOTOS_FOLDER, '{}_out_photo.jpg'.format(str(update.message.from_user.username)))
    photo_file.download(path_style)
    logger.info("Photo of %s: %s", user.first_name, 'style_photo.jpg')
    update.message.reply_text('Перенос стиля скоро будет⏳')
    style_img = model.image_loader(path_style)
    content_img = model.image_loader(path_content)
    output = model.run_style_transfer(content_img, style_img, content_img)
    imsave(output, path_out)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(path_out, 'rb'))
    update.message.reply_text('По-моему получилось неплохо👨‍🎨')
    imdel(path_style)
    imdel(path_content)
    imdel(path_out)
    update.message.reply_text('Хочешь еще?\nШли новое фото📷.\nВведи /cancel для завершения')
    return CONTENT

@run_async
def cancel(update, context):
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    path_style = os.path.join(PHOTOS_FOLDER, '{}_style_photo.jpg'.format(str(update.message.from_user.username)))
    path_content = os.path.join(PHOTOS_FOLDER, '{}_content_photo.jpg'.format(str(update.message.from_user.username)))
    path_out = os.path.join(PHOTOS_FOLDER, '{}_out_photo.jpg'.format(str(update.message.from_user.username)))
    imdel(path_style)
    imdel(path_content)
    imdel(path_out)
    
    update.message.reply_text('Если захочешь попробовать еще, просто введи /start')
    

    return ConversationHandler.END


@run_async
def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

if __name__ == '__main__':
    
    from telegram.ext import Updater, MessageHandler, Filters, ConversationHandler, CommandHandler
    import logging
    if not os.path.exists(PHOTOS_FOLDER):
        os.makedirs(PHOTOS_FOLDER)
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

            STYLE: [MessageHandler(Filters.photo,photo_style)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)

    # log all errors
    dp.add_error_handler(error)

    updater.start_polling()

