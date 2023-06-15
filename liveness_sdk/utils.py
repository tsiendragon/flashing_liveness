from datetime import datetime


def create_datetime_str():
    date_time_obj = datetime.now()
    timestamp_str = date_time_obj.strftime("%d-%b-%Y (%H:%M:)")
    return timestamp_str


def square_crop_image(image):
    imgh, imgw = image.shape[:2]
    if imgh > imgw:
        image = image[(imgh - imgw) // 2 : (imgh + imgw) // 2, :]  # NOQA
    else:
        image = image[:, (imgw - imgh) // 2 : (imgw + imgh) // 2]  # NOQA
    return image
