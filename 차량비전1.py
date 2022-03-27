from google.colab.patches import cv2_imshow
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import drive

def saturation(value):
  if(value>255):
    value = 255
  return value

def wrapping(value):
  while(value > 255):
    value = value - 256
  return value

#gray_img 그레이스케일 이미지, img1,2는 이진이미지 변환용
gray_img = cv2.imread('/content/gdrive/MyDrive/Vision/bridge_gray.bmp')
img1 = cv2.imread('/content/gdrive/MyDrive/Vision/bridge.bmp')
img2 = cv2.imread('/content/gdrive/MyDrive/Vision/bridge.bmp')

#이진영상 output
and_output_img = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
or_output_img = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

#output img 곱셈연산 output_img1 나눗셈 연산 gray_img1 그레이스케일로 변환
output_img = np.zeros((gray_img.shape[0],gray_img.shape[1]),dtype=np.ubyte)
output_img1 = np.zeros((gray_img.shape[0],gray_img.shape[1]),dtype=np.ubyte)
gray_img1 = np.zeros((gray_img.shape[0],gray_img.shape[1]),dtype=np.ubyte)

all_add_img=np.zeros((gray_img.shape[0],gray_img.shape[1]),dtype=np.ubyte)
all_sat_add_img=np.zeros((gray_img.shape[0],gray_img.shape[1]),dtype=np.ubyte)
all_wrap_add_img=np.zeros((gray_img.shape[0],gray_img.shape[1]),dtype=np.ubyte)

#plus output img
plus_output_img = np.zeros((gray_img.shape[0],gray_img.shape[1]),dtype=np.ubyte)

#saturation, wrapping을 적용할 output img2,3
output_img2 = np.zeros((gray_img.shape[0],gray_img.shape[1]),dtype=np.ubyte)
output_img3 = np.zeros((gray_img.shape[0],gray_img.shape[1]),dtype=np.ubyte)

#그레이스케일 영상으로 변환
for h in range(gray_img.shape[0]):
  for w in range(gray_img.shape[1]):
    gray_img1[h,w] = 0.2126*gray_img[h,w,0]+0.7152*gray_img[h,w,0]+0.0722*gray_img[h,w,0]

#RGB img 크기 192x128
RGB_img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
RGB_img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

R_img1,G_img1,B_img1 = cv2.split(RGB_img1)
R_img2,G_img2,B_img2 = cv2.split(RGB_img2)

R_AND=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
G_AND=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
B_AND=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)

R_OR=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
G_OR=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
B_OR=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)

#영상 이진화
for h in range(RGB_img1.shape[0]):
  for w in range(RGB_img1.shape[1]):
    if(np.int32(R_img1[h,w])>180):
      R_img1[h,w]= G_img1[h,w]=B_img1[h,w] = 255
    else:
      R_img1[h,w]= G_img1[h,w]=B_img1[h,w] = 0
    if(np.int32(G_img2[h,w])>50):
      R_img2[h,w]= G_img2[h,w]=B_img2[h,w] = 255
    else:
      R_img2[h,w]= G_img2[h,w]=B_img2[h,w] = 0

#saturation 적용X
#곱셈
for h in range(gray_img1.shape[0]):
  for w in range(gray_img1.shape[1]):
    output_img[h,w] = (np.int32(gray_img1[h,w]*1.3))

#나눗셈
for h in range(gray_img1.shape[0]):
  for w in range(gray_img1.shape[1]):
    output_img1[h,w] = np.fabs(np.float32(gray_img1[h,w]/1.3))
#덧셈
for h in range(gray_img1.shape[0]):
  for w in range(gray_img1.shape[1]):
    plus_output_img[h,w] = (np.float32(gray_img1[h,w]+200))

#이전결과 다 더하기
for h in range(gray_img1.shape[0]):
  for w in range(gray_img1.shape[1]):
    all_add_img[h,w] = np.int32(output_img[h,w])+ np.float32(output_img1[h,w])+np.float32(plus_output_img[h,w])

#이전결과 sat
for h in range(gray_img1.shape[0]):
  for w in range(gray_img1.shape[1]):
    all_sat_add_img[h,w] = saturation(np.int32(output_img[h,w])+ np.float32(output_img1[h,w])+np.float32(plus_output_img[h,w]))

#이전결과 wrap
for h in range(gray_img1.shape[0]):
  for w in range(gray_img1.shape[1]):
    all_wrap_add_img[h,w] = wrapping(np.int32(output_img[h,w])+ np.float32(output_img1[h,w])+np.float32(plus_output_img[h,w]))

#saturation 적용O
for h in range(gray_img1.shape[0]):
  for w in range(gray_img1.shape[1]):
    output_img2[h,w] = saturation(np.float32(gray_img1[h,w]+200))
#wrapping 적용O
for h in range(gray_img1.shape[0]):
  for w in range(gray_img1.shape[1]):
    output_img3[h,w] = wrapping(np.int32(gray_img1[h,w]+200))
#이진영상 AND
for h in range(RGB_img1.shape[0]):
  for w in range(RGB_img1.shape[1]):
    R_AND[h,w] = saturation(np.int32(R_img1[h,w]) & np.int32(R_img2[h,w]))
    G_AND[h,w] = saturation(np.int32(G_img1[h,w]) & np.int32(G_img2[h,w]))
    B_AND[h,w] = saturation(np.int32(B_img1[h,w]) & np.int32(B_img2[h,w]))

#이진영상 OR
for h in range(RGB_img1.shape[0]):
  for w in range(RGB_img1.shape[1]):
    R_OR[h,w] = saturation(np.int32(R_img1[h,w]) | np.int32(R_img2[h,w]))
    G_OR[h,w] = saturation(np.int32(G_img1[h,w]) | np.int32(G_img2[h,w]))
    B_OR[h,w] = saturation(np.int32(B_img1[h,w]) | np.int32(B_img2[h,w]))

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.title("multiply Calc")
plt.imshow(output_img)
plt.axis("off")

plt.figure(figsize=(20,20))
plt.subplot(1,2,2)
plt.title("divide Calc")
plt.imshow(output_img1)
plt.axis("off")

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.title("plus+200")
plt.imshow(plus_output_img)
plt.axis("off")

plt.figure(figsize=(20,20))
plt.subplot(2,2,2)
plt.title("plus saturation")
plt.imshow(output_img2)
plt.axis("off")

plt.figure(figsize=(20,20))
plt.subplot(3,2,1)
plt.title("plus wrapping")
plt.imshow(output_img3)
plt.axis("off")

plt.figure(figsize=(20,20))
plt.subplot(3,2,2)
plt.title("add gray+multiply+divide")
plt.imshow(all_add_img)
plt.axis("off")

plt.figure(figsize=(20,20))
plt.subplot(4,2,1)
plt.title("sat add gray+multiply+divide")
plt.imshow(all_sat_add_img)
plt.axis("off")

plt.figure(figsize=(20,20))
plt.subplot(4,2,2)
plt.title("wrap add gray+multiply+divide")
plt.imshow(all_wrap_add_img)
plt.axis("off")

RGB_img1[:,:,0] =R_img1
RGB_img1[:,:,1] =G_img1
RGB_img1[:,:,2] =G_img1
RGB_img2[:,:,0] =R_img2
RGB_img2[:,:,1] =G_img2
RGB_img2[:,:,2] =B_img2

plt.figure(figsize=(20,20))
plt.subplot(5,2,1)
plt.title("Binary img1")
plt.imshow(RGB_img1)
plt.axis("off")

plt.figure(figsize=(20,20))
plt.subplot(5,2,2)
plt.title("Binary img2")
plt.imshow(RGB_img2)
plt.axis("off")

and_output_img[:,:,0] = R_AND
and_output_img[:,:,1] = G_AND
and_output_img[:,:,2] = B_AND
plt.figure(figsize=(20,20))
plt.subplot(6,2,1)
plt.title("AND")
plt.imshow(and_output_img)
plt.axis("off")

or_output_img[:,:,0] = R_OR
or_output_img[:,:,1] = G_OR
or_output_img[:,:,2] = B_OR
plt.figure(figsize=(20,20))
plt.subplot(6,2,2)
plt.title("OR")
plt.imshow(or_output_img)
plt.axis("off")
