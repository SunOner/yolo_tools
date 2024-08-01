import asyncio
from urllib.request import Request, urlopen
import tempfile
from PIL import Image, ImageSequence
from moviepy.editor import VideoFileClip
import cv2
import discord
from discord.ext import commands
import numpy as np
from ultralytics import YOLO
from io import BytesIO

# discord developer: https://discord.com/developers/applications
##################################### SETTINGS #######################################
discord_token = ''
discord_channel = 'auto_annotate'
discord_max_file_size = 500000000 # 500 MB
discord_max_pixels_size = [4000, 4000] # weight, height

AI_model = 'models/sunxds_0.4.1.engine'
AI_verbose = False
AI_image_size = 480
AI_device = 0
AI_classes = range(11) # use raw array [0, 1, 2, 3, 4, 5, 6, 7, 8] or range(max_model_classes)
AI_conf = 0.20

app_workers = 4 # max thread workers

######################################################################################
model = YOLO(AI_model, task='detect') # init model

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

async def process_gif(image_url):
    req = Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urlopen(req)
    gif = Image.open(BytesIO(resp.read()))
    
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    annotated_frames = []
    
    for frame in frames:
        open_cv_image = np.array(frame)
        if open_cv_image.ndim == 3:
            if open_cv_image.shape[2] == 4:
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGRA2BGR)
            else:
                open_cv_image = open_cv_image[:, :, ::-1].copy()
        elif open_cv_image.ndim == 2:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2RGB)
        
        h, w, _ = open_cv_image.shape
        results = model.predict(source=open_cv_image, conf=AI_conf, device=AI_device, classes=AI_classes, verbose=AI_verbose, imgsz=AI_image_size)
        
        bboxes_ = results[0].boxes.xyxy.tolist()
        for index, bbox in enumerate(bboxes_):
            color = (0, 0, 255)
            xmin, ymin, xmax, ymax = list(map(int, bbox))
            cv2.rectangle(open_cv_image, (xmin, ymin), (xmax, ymax), color, 2)
            class_index = results[0].boxes.cls[index].item()
            class_name = results[0].names[class_index]
            conf = results[0].boxes.conf[index]
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(open_cv_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        annotated_frames.append(Image.fromarray(open_cv_image))
    
    io_buf = BytesIO()
    annotated_frames[0].save(
        io_buf, format='GIF', save_all=True, 
        append_images=annotated_frames[1:], loop=1, duration=gif.info['duration']
    )
    io_buf.seek(0)
    return io_buf

async def process_video(video_url):
    req = Request(video_url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urlopen(req)
    video_data = resp.read()
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_path = f'{tmpdirname}/input.mp4'
        output_path = f'{tmpdirname}/output.mp4'
        
        with open(input_path, 'wb') as video_file:
            video_file.write(video_data)
        
        clip = VideoFileClip(input_path)
        processed_clip = clip.fl_image(process_frame)
        
        processed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        with open(output_path, 'rb') as f:
            video_io = BytesIO(f.read())
            
        clip.close()
    return video_io

def process_frame(frame):
    frame = np.array(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
    
    results = model.predict(source=frame_rgb, conf=AI_conf, device=AI_device, classes=AI_classes, verbose=AI_verbose, imgsz=AI_image_size)

    bboxes_ = results[0].boxes.xyxy.tolist()
    confs_ = results[0].boxes.conf.tolist()
    classes_ = results[0].boxes.cls.tolist()
    cls_dict = results[0].names
    
    for index, bbox in enumerate(bboxes_):
        color = (0, 255, 0)
        xmin, ymin, xmax, ymax = list(map(int, bbox))
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        class_name = cls_dict[classes_[index]]
        conf = confs_[index]
        label = f"{class_name}: {conf:.2f}"
        
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

async def process_image(image_url):
    req = Request(
        image_url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    )
    resp = urlopen(req)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    h, w, _ = image.shape
    
    results = model.predict(source=image, conf=AI_conf, device=AI_device, classes=AI_classes, verbose=AI_verbose, imgsz=AI_image_size)

    bboxes_ = results[0].boxes.xyxy.tolist()
    bboxes = list(map(lambda x: list(map(lambda y: int(y), x)), bboxes_))
    confs_ = results[0].boxes.conf.tolist()
    confs = list(map(lambda x: int(x*100), confs_))
    classes_ = results[0].boxes.cls.tolist()
    classes = list(map(lambda x: int(x), classes_))
    cls_dict = results[0].names
    class_names = list(map(lambda x: cls_dict[x], classes))
    
    annot_lines = []
    for index, val in enumerate(class_names):
        xmin, ymin, xmax, ymax = int(bboxes[index][0]), int(bboxes[index][1]), int(bboxes[index][2]), int(bboxes[index][3])
        width = xmax - xmin
        height = ymax - ymin
        center_x = xmin + (width/2)
        center_y = ymin + (height/2) 
        annotation = f"{classes[index]} {center_x/w} {center_y/h} {width/w} {height/h}"
        annot_lines.append(annotation)

    for index, bbox in enumerate(bboxes):
        color = (0, 0, 255)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label = f"{class_names[index]}: {confs_[index]:.2f}"
        cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        raise Exception("Could not encode image!")
    io_buf = BytesIO(buffer)
    return annot_lines, io_buf
    
queue = asyncio.Queue()

async def worker():
    while True:
        task = await queue.get()
        try:
            await process_task(task)
        finally:
            queue.task_done()

async def process_task(task):
    message, attachment = task
    if attachment.height is not None and attachment.width is not None:
        if attachment.height <= discord_max_pixels_size[1] and attachment.width <= discord_max_pixels_size[0]:
            print(f'New message: Content-type: {attachment.content_type}')
            if attachment.size <= discord_max_file_size:
                if attachment.content_type in ['image/jpeg', 'image/jpg', 'image/png']:
                    image_url = attachment.url
                    annot_lines, image_io = await process_image(image_url)
                    if len(annot_lines) >= 1:
                        final_message = '```\n'
                        for line in annot_lines:
                            final_message += f'{line}\n'
                        final_message += '```'
                        file = discord.File(fp=image_io, filename="image.jpg")
                        await message.reply(final_message, file=file)
                    else:
                        await message.reply('No detections')
                        
                if attachment.content_type in ['video/mp4', 'video/avi']:
                    video_url = attachment.url
                    video_io = await process_video(video_url)
                    file = discord.File(fp=video_io, filename="video.mp4")
                    await message.reply(file=file)
                
                if attachment.content_type in ['image/gif']:
                    image_url = attachment.url
                    image_io = await process_gif(image_url)
                    file = discord.File(fp=image_io, filename="image.gif")
                    await message.reply(file=file)
            else:
                await message.reply(f'The content are too large.')
        else:
            await message.reply('The images are too large.')

@bot.event
async def on_message(message):
    if message.channel.name != discord_channel or message.author == bot.user or not message.attachments:
        return
    for attachment in message.attachments:
        await queue.put((message, attachment))

async def setup_workers():
    for _ in range(app_workers):
        bot.loop.create_task(worker())

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    await setup_workers()

bot.run(discord_token)