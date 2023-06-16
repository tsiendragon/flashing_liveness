import cv2
import numpy as np

# Load the image

if False:
    image = cv2.imread('./data/r/red.png')
    image2 = cv2.imread('./data/r/white.png')
    image3 = cv2.imread('./data/r/blue.png')
    image4 = cv2.imread('./data/r/green.png')
else:
    image = cv2.imread('./data/f/red.png')
    image2 = cv2.imread('./data/f/white.png')
    image3 = cv2.imread('./data/f/blue.png')
    image4 = cv2.imread('./data/f/green.png')


print(image.shape)

image=cv2.resize(image,(224,224))
image2=cv2.resize(image2,(224,224))
image3=cv2.resize(image3,(224,224))
image4=cv2.resize(image4,(224,224))


image=image.astype(np.float32)
image2=image2.astype(np.float32)
image3=image3.astype(np.float32)
image4=image4.astype(np.float32)



#Split the image into its color channels
b, g, r= cv2.split(image)
b2, g2, r2= cv2.split(image2)
b3, g3, r3= cv2.split(image3)
b4, g4, r4= cv2.split(image4)


x=r3-r2
w=b3-b2
z=g3-g2




y=image-image2


x=np.clip(x,0,255)

w=np.clip(w,0,255)

z=np.clip(z,0,255)

h,v = x.shape

print(h,v)
h=h//3
v=v//3

x_new=z[h:2*h,v:2*v]
z_new=w[h:2*h,v:2*v]


print(np.mean(x_new+z_new))
print(np.mean(w))
print(np.mean(z))

#print((r-r2).astype(int))
np.savetxt('big_matrix.txt',w,fmt='%d',delimiter=' ')

# Set the red and green channels to zero (keeping only the blue channel)
zero_channel = np.zeros_like(b)
blue_image = cv2.merge((zero_channel,z,x))

black=np.zeros((224,224,3), dtype=np.uint8)
white=black-1
white = np.clip(white, a_min=0, a_max=None)

# Display or save the blue channel image
cv2.imshow('Blue Channel',blue_image)





cv2.waitKey(0)
cv2.destroyAllWindows()