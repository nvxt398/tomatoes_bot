import torch
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import logging
from process_leaf import segment_leaf, load_model

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

model = load_model("model_best_weights.pth")

classes = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

classes_rus = {
    'Tomato___Bacterial_spot': 'Бактериальная пятнистость',
    'Tomato___Early_blight': 'Ранний фитофтороз',
    'Tomato___Late_blight': 'Поздний фитофтороз',
    'Tomato___Leaf_Mold': 'Плесень листьев',
    'Tomato___Septoria_leaf_spot': 'Септориоз',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Поражение паутинным клещом',
    'Tomato___Target_Spot': 'Точечная пятнистость',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Желтая курчавость листьев',
    'Tomato___Tomato_mosaic_virus': 'Вирус мозаики томата',
    'Tomato___healthy': 'Здоровое растение'
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Отправьте фотографию листа томата для определения заболевания")

async def photo_processing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    await file.download_to_drive("image.jpg")

    img = Image.open("image.jpg").convert("RGB")
    segmented_img = segment_leaf(img)
    img_tensor = transform(segmented_img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        result = classes_rus.get(classes[predicted.item()])

    await update.message.reply_text(f"Наиболее вероятный диагноз: {result}")

async def main():
    token = ""  # вставь токен сюда
    app = ApplicationBuilder().token(token).read_timeout(30).connect_timeout(30).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, photo_processing))
    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(main())

