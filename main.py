from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import time
import os
import json
import redis
import traceback
start_time = time.clock()
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

env = os.environ['HOME']
def connect_to_redis(counter=0):
    if env == 'production':
        r = redis.Redis(host = '91.240.87.81', port = 6379, db = 10, password = '3P5SPRiOwYd1ZrUqaEbb7cqZgdbow')
    else:
        r = redis.Redis(host = 'localhost', port = 6379, db = 10)
    if r.ping():
        return r
    else:
        if counter > 20:
            sys.exit()
        time.sleep( 1 )
        connect_to_redis(counter+1)
    

r=connect_to_redis()
print(r)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)


model = load_model( MODEL_FILENAME )
def answer(meta,result,error):
    result = dict(error=error,
        result=result,
        meta=meta)
    r.set(meta["name"],json.dumps( result ))

while True:
    if r.ping()!=True:
        r=connect_to_redis()
    if r.exists('captchas'):
        next_captcha = r.blpop( 'captchas' )
        print( "Найдено задание" )
        print( next_captcha )
    else:
        next_captcha = False
    if next_captcha == False:
        time.sleep( 0.5 )
        continue
    try:
        current_captcha_json = json.loads(next_captcha[1].decode("utf-8"))
    except Exception:
        formated_error = traceback.format_exception()
        print(formated_error)
        print(next_captcha[1].decode("utf-8"))
        print("Не могу распарсить данные json из массива captchas")
        print("Обработки нет, результата нет, нечего писать")
        continue
    #file path абсолютный /...
    image_file=current_captcha_json["file"]
    #мета будет записана как есть
    meta=current_captcha_json["meta"]
    # флаг ошибки обработки
    error=False
    # имя файла без сети
    name = image_file.split("/")[-1].split(".")[0]
    # Load the image and convert it to grayscale
    print(image_file)
    image = cv2.imread(image_file)
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        
        print("`{}` Ошибка конвертации в grayscale".format(name))
        print("Попытка работать с файлом как есть...")


    
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))
    regions_c = len(letter_image_regions)
    if regions_c != 6:
        print("Не удалось выделить 6 областей для распознавания файл не будет обработан")
        answer(meta,"","Expected 6 regions, detected: {}".format(regions_c))
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)
    predictions = []

    # loop over the lektters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # Re-size the letter image to 20x20 pixels to match training data
        try:
            letter_image = resize_to_fit(letter_image, 20, 20)
        except:
            print ("Не возможно изменить размер региона, ошибка операции")
            answer(meta,"","Dont make new size for region, operation error"+str(regions_c))
            continue

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    answer(meta,captcha_text,"")

