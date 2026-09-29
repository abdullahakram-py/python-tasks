import cv2
print('cv2 file:', cv2.__file__)
print('cv2 version:', getattr(cv2, '__version__', 'unknown'))
print('has CascadeClassifier', hasattr(cv2, 'CascadeClassifier'))
print('has data', hasattr(cv2, 'data'))
cv2_data = getattr(cv2, 'data', None)
if cv2_data is not None:
    print('cv2.data attrs:', [n for n in dir(cv2_data) if 'haar' in n.lower() or 'cascade' in n.lower()])
else:
    print('cv2.data is None')
