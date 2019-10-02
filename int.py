#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import cv2
import json
import numpy as np
import traceback

import darknet.python.darknet as dn
from darknet.python.darknet import detect

from os import makedirs
from os.path import splitext, basename, dirname, isdir, isfile

from src.utils import crop_region, image_files_from_folder, im2single, nms
from src.label import Shape, readShapes, writeShapes, dknet_label_conversion, Label, lread, lwrite
from src.keras_utils import load_model, detect_lp
from src.drawing_utils import draw_label, draw_losangle, write2img

import os
import logging
import exifread
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
YELLOW = (  0,255,255)
RED    = (  0,  0,255)

# structure: dict of lists of dict {[{}]}
plates_db = {}
with open('plates_all.json', 'r') as json_file:
    data = json.load(json_file)
    for plate in data["plates"]:
        if plate["plate"] in plates_db:
            plates_db[plate["plate"]].append(plate)
        else:
            plates_db[plate["plate"]] = [plate]

help_msg = "使用方法:\n1. send一張有車及車牌既圖片，我會自動識車牌及查詢相關資料\n2. 查詢車牌，例如: /check ABC123\n3. 查詢地區 /area 柴灣"

def format_msg(p):
    if p["source"] == "TG: @dadfindcar":
        return "車牌: {}\n牌子: {}\n型號: {}\n顏色: {}\n出沒地址: {}\n詳細說明: {}\n類別: {}\n編號: {}\nSource: {}\n\n".format(p["plate"], p["make"], p["model"], p["color"].encode('utf-8'), p["location"].encode('utf-8'), p["comment"].encode('utf-8'), p["category"].encode('utf-8'), p["ID"], p["source"])
    else:
        return "車牌: {}\n詳細說明: {}\nSource: {}\n\n".format(p["plate"], p["description"].encode('utf-8'), p["source"])
    return ""

def find_plate(target):
    text = ""
    # preprocess, plates always upper case, do not contain "I", "O" or "Q"
    target = target.upper()
    target = target.replace("I", "1")
    target = target.replace("O", "0")
    target = target.replace("Q", "0")

    print("target plate:", target)
    if target in plates_db: 
        for p in plates_db[target]:
            text = text + format_msg(p)
    else:
        text = "NO Record!"

    return text

# prepare vehicle detection model
vehicle_threshold = .5
vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
vehicle_dataset = 'data/vehicle-detector/voc.data'
vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
vehicle_meta = dn.load_meta(vehicle_dataset)

# prepare LP detection model
lp_threshold = .5
wpod_net_path = 'data/lp-detector/wpod-net_update1.h5'
#wpod_net = load_model(wpod_net_path)

# perpare OCR model
ocr_threshold = .4
ocr_weights = 'data/ocr/ocr-net.weights'
ocr_netcfg  = 'data/ocr/ocr-net.cfg'
ocr_dataset = 'data/ocr/ocr-net.data'
ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
ocr_meta = dn.load_meta(ocr_dataset)

def genOutput(img_path, output_dir, bname):
    I = cv2.imread(img_path)
    detected_cars_labels = '%s/%s_cars.txt' % (output_dir,bname)
    
    Lcar = lread(detected_cars_labels)
    
    sys.stdout.write('%s' % bname)
    
    if Lcar:
        for i,lcar in enumerate(Lcar):
            draw_label(I,lcar,color=YELLOW,thickness=3)
            
            lp_label = '%s/%s_%dcar_lp.txt' % (output_dir,bname,i)
            lp_label_str = '%s/%s_%dcar_lp_str.txt' % (output_dir,bname,i)
            
            if isfile(lp_label):
                Llp_shapes = readShapes(lp_label)
                pts = Llp_shapes[0].pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
                ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
                draw_losangle(I,ptspx,RED,3)
                
                if isfile(lp_label_str):
                    with open(lp_label_str,'r') as f:
                        lp_str = f.read().strip()
                    llp = Label(0,tl=pts.min(1),br=pts.max(1))
                    write2img(I,llp,lp_str)
                    
                    sys.stdout.write(',%s' % lp_str)
    cv2.imwrite('%s/%s_output.png' % (output_dir,bname),I, [int(cv2.IMWRITE_PNG_COMPRESSION),9])

def OCRDection(LPImagePath):
    # OCR recognition
    W,(width,height) = detect(ocr_net, ocr_meta, LPImagePath ,thresh=ocr_threshold, nms=None)
    if len(W):
        L = dknet_label_conversion(W,width,height)
        L = nms(L,.45)
        L.sort(key=lambda x: x.tl()[0])
        lp_str = ''.join([chr(l.cl()) for l in L])

        # plates always upper case, do not contain "I", "O" or "Q"
        lp_str = lp_str.upper()
        lp_str = lp_str.replace("I", "1")
        lp_str = lp_str.replace("O", "0")
        lp_str = lp_str.replace("Q", "0")

        bname = basename(splitext(LPImagePath)[0])
        dname = dirname(LPImagePath)
        LPTextPath = '%s/%s_str.txt' % (dname, bname)

        with open(LPTextPath, 'w') as f:
            f.write(lp_str + '\n')
        return lp_str
    else:
        return ""

def LPDection(carImagePath):
    wpod_net = load_model(wpod_net_path)
    print carImagePath
    Ivehicle = cv2.imread(carImagePath)
    ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
    side  = int(ratio*288.)
    bound_dim = min(side + (side%(2**4)),608)

    Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

    if len(LlpImgs):
        bname = basename(splitext(carImagePath)[0])
        dname = dirname(carImagePath)
        Ilp = LlpImgs[0]
        cv2.imwrite('%s/%s_lp_raw.png' % (dname, bname), Ilp*255.)
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

        s = Shape(Llp[0].pts)

        LPImagePath = '%s/%s_lp.png' % (dname, bname)
        cv2.imwrite(LPImagePath, Ilp*255.)
        LPTextPath = '%s/%s_lp.txt' % (dname, bname)
        writeShapes(LPTextPath, [s])
        return LPImagePath
    else:
        return ""

def vehicle_detection(img_path, output_dir):
    try:
        if not isdir(output_dir):
            makedirs(output_dir)

        bname = basename(splitext(img_path)[0])
        plates = []

        # Vehicle detection
        R,_ = detect(vehicle_net, vehicle_meta, img_path ,thresh=vehicle_threshold)
        R = [r for r in R if r[0] in ['car','bus', 'motorbike']]

        # print '\t\t%d cars found' % len(R)
        if not len(R):
            return ("", plates)

        Iorig = cv2.imread(img_path)
        WH = np.array(Iorig.shape[1::-1],dtype=float)
        Lcars = []

        for i,r in enumerate(R):
            cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
            tl = np.array([cx - w/2., cy - h/2.])
            br = np.array([cx + w/2., cy + h/2.])
            label = Label(0,tl,br)
            Icar = crop_region(Iorig,label)

            Lcars.append(label)
            carImagePath = '%s/%s_%dcar.png' % (output_dir,bname,i)
            cv2.imwrite(carImagePath, Icar)
            #print("CarImagePath: ", carImagePath)

            # LP detection
            LPImagePath = LPDection(carImagePath)
            if LPImagePath:
                 lp_str = OCRDection(LPImagePath)
                 if lp_str:
                     plates.append(lp_str)

        lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)

        # draw yellow box around the cars and red box around license plates
        genOutput(img_path, output_dir, bname)
        return ('%s/%s_output.png' % (output_dir,bname), plates)

    except:
        traceback.print_exc()
        return ("", plates)

download_path = os.path.dirname(os.path.realpath(__file__)) + "/"

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def process_exif(file_name):
    if os.path.exists(file_name):
       with open(file_name, 'rb') as ff:
           try:
               # do stuff
               print("processing EXIF")
               tags = exifread.process_file(ff)
               for tag in tags.keys():
                   print("Key: %s, value %s" % (tag, tags[tag]))
           except : # whatever reader errors you care about
               # handle error
               print("open exception")

def start(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text=help_msg)

def help(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text=help_msg)

def find_area(area):
    text = ""
    for plate in plates_db:
        for record in plates_db[plate]:
            if "location" in record:
                if record["location"] and area in record["location"]:
                     text = text + format_msg(record)
    return text if len(text) > 0 else "No Record!"

def area(update, context):
    token = update.message.text.split(" ")
    if len(token) < 2:
        text = "Command format Error!"
    else:
        text = find_area(token[1])
    context.bot.send_message(chat_id=update.message.chat_id, text=text)

def check(update, context):
    token = update.message.text.split(" ")
    if len(token) < 2:
        text = "Command format Error!"
    else:
        text = find_plate(token[1])
    context.bot.send_message(chat_id=update.message.chat_id, text=text)

def photo(update, context):
    output_dir = download_path + "/output/"

    context.bot.send_message(chat_id=update.message.chat_id, text="photo received.")
    biggest = -1
    index = -1
    for i, f in enumerate(update.message.photo):
        if f.file_size >= biggest:
              index = i
    if index != -1:
        file_name = f.get_file().download()
        img_path = download_path + file_name
        (output_file, plates) = vehicle_detection(img_path, output_dir)
        if output_file:
            context.bot.send_photo(chat_id=update.message.chat_id, photo=open(output_file, 'rb'))
        if len(plates):
            for plate in plates:
                context.bot.send_message(chat_id=update.message.chat_id, text=find_plate(plate))
                 
        #process_exif(download_path + file_name):

def document(update, context):
    output_dir = download_path + "/output/"
    context.bot.send_message(chat_id=update.message.chat_id, text="document received.")

    file_name = update.message.document.get_file().download()
    img_path = download_path + file_name
    #process_exif(download_path + file_name)
    (output_file, plates) = vehicle_detection(img_path, output_dir)
    if output_file:
        context.bot.send_photo(chat_id=update.message.chat_id, photo=open(output_file, 'rb'))

def unknownCMD(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="Sorry, I didn't understand that command.")

def textMsg(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="Sorry, I am a bot, I don't understand your message.")

if __name__ == '__main__':
    # get TG key from env variable
    token = os.environ.get('TG_KEY')

    updater = Updater(token, use_context=True)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)
    
    check_handler = CommandHandler('check', check)
    dispatcher.add_handler(check_handler)

    area_handler = CommandHandler('area', area)
    dispatcher.add_handler(area_handler)

    help_handler = CommandHandler('help', help)
    dispatcher.add_handler(help_handler)

    photo_handler = MessageHandler(Filters.photo, photo)
    dispatcher.add_handler(photo_handler)
    
    unknownCMD_handler = MessageHandler(Filters.command, unknownCMD)
    dispatcher.add_handler(unknownCMD_handler)
    
    textMsg_handler = MessageHandler(Filters.text, textMsg)
    dispatcher.add_handler(textMsg_handler)
    
    updater.start_polling()

    
