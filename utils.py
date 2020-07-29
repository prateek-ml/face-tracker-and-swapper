import cv2 
import numpy as np

def isGray(image):
    """Return True is image has one channel per pixel"""
    return image.ndim < 3

def DividedWidthHeight(image, divisor):
    """Return an image's dimensions divided by a value"""
    h, w = image.shape[:2]
    return (w//divisor, h//divisor)