import tensorflow as tf
import cv2
from scipy.ndimage.filters import gaussian_filter
from time import clock
import numpy as np

import pyautogui as pg


############################### tuning #############################
#accuracy boosting variables
mul=1 # 2 for more accuracy but less speed
steps=1 # 1 for more accuracy but less speed , 4 for more post processing speed but less accuracy
visual =True #False; visualising takes post processing time
#sensistivity of mousec 0-1-2
kx=0.5 
ky=1
###############################################################################
KERAS_MODEL_FILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\my_hand_openpose_tf\\hand_model.h5"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


(screen_sizex,screen_sizey)=pg.size()

# Or loading from prepared weigths file

model=tf.keras.models.load_model(KERAS_MODEL_FILE)
print('LOADED!')

def find_peaks(heatmap_avg, thre=0.1, sigma=3):
    all_peaks = []
    peak_counter = 0

    #for part in range(0, heatmap_avg.shape[-1],steps):
    for part in [8,12]:
    
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=sigma)


        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > thre))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        peaks_with_score_1=[]
        for k in peaks_with_score:
            if k[2]>0.2:
                peaks_with_score_1.append(k+(part,))
        #id = range(peak_counter, peak_counter + len(peaks))
        #peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        #all_peaks.append(peaks_with_score_and_id)
        all_peaks.append(peaks_with_score_1)
        peak_counter += len(peaks)
    return all_peaks, peak_counter


def parts(all_peaks):
    partx = np.ones(22)*-1
    party = np.ones(22)*-1
    for i in range(len(all_peaks)):
        for j in all_peaks[i]:
            partx[j[3]] = j[0]
            party[j[3]] = j[1]
    return  partx,party      
def process_image(model, image_orig, scale_mul=1, peaks_th=0.1, sigma=3, mode='heatmap'):
    
    scale = 368/image_orig.shape[1]
    scale = scale*scale_mul
    image =  cv2.resize(image_orig, (0,0), fx=scale, fy=scale) 

    start = clock()
    net_out = model.predict(np.expand_dims( image /256 -0.5 ,0))
    stop = clock()
    took = stop-start
    
    out = cv2.resize( net_out[0], (image_orig.shape[1], image_orig.shape[0]) )
    image_out = image_orig
    #print("image shape",(image_orig.shape[1], image_orig.shape[0]))
    center_y,center_x=None,None
    
    mask = np.zeros_like(image_out).astype(np.float32)
    if mode == 'heatmap':
        #0->22-2
        for chn in range(0, out.shape[-1]-2):
            m = np.repeat(out[:,:,chn:chn+1],3, axis=2)
            m = 255*( np.abs(m)>0.2)
            
            mask = mask + m*(mask==0)
        mask = np.clip(mask, 0, 255)
        image_out = image_out*0.8 + mask*0.2
    else:
        peaksR = find_peaks(out, peaks_th, sigma=sigma)[0]
        print("-"*10)
        print(peaksR)    
        #print(peaksR,len(peaksR))
        #print(parts(peaksR))
        px,py=parts(peaksR)
        #peaksL = find_peaks(-out, peaks_th, sigma=sigma)[0]
        peak1=[]
        peak2=[]
        
        for peak in peaksR:

            if(len(peak)):
                peak = peak[0]
                #cv2.drawMarker(image_out, (peak[0], peak[1]), (0,0,255), cv2.MARKER_STAR )
                peak1.append(peak[0])
                peak2.append(peak[1])
        if(len(peak1)):    
            peak1=np.array(peak1)
            peak2=np.array(peak2)        
            center_x=np.sum(peak1)//len(peak1)
            center_y=np.sum(peak2)//len(peak2)

            #cv2.drawMarker(image_out, (int(center_x),int(center_y)), (255,0,0), cv2.MARKER_STAR )
        """
        for peak in peaksL:
            if(len(peak)):
                peak = peak[0]
                cv2.drawMarker(image_out, (peak[0], peak[1]), (255,0,0), cv2.MARKER_STAR )
        """        
                
    image_out = np.clip(image_out, 0, 255).astype(np.uint8)
                
    return image_out, took,center_x,center_y,px,py

x=0
y=0

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
#model=keras.models.load_model(r"C:\\Users\\Dell\\Desktop\\holidASY\\openpose_hand_keras\\openpose_hand_keras.h5")
#model.summary()
while 1:
    ret, image = cap.read()
    image=cv2.flip(image,1)
    #clear_output()

    image_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start = clock()
    image_out, inference_took,X,Y,xs,ys = process_image(model, image,mode='none')
    frameWidth,frameHeight,channel=np.shape(image_out)
    #print(xs,ys)
    for i in range(22):
        if xs[i]>=0:
            cv2.putText(image_out,str(i),(int(xs[i]),int(ys[i])), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255),1, lineType=cv2.LINE_AA)
    #left middle finger controls the mouse , bringing left index close to left middle will click
    X,Y = int(xs[12]),int(ys[12])#12
    X2,Y2 = int(xs[8]),int(ys[8])
    if X>0 and X2>0:
        dist = (X-X2)**2 + (Y-Y2)**2
        cv2.putText(image_out,str(dist),(10,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255),1, lineType=cv2.LINE_AA)
        if dist <800:
            cv2.putText(image_out,"click",(X2,Y2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255),1, lineType=cv2.LINE_AA)
            print("____click____")
            pg.click()
    if X<0:
        X,Y=frameWidth//2,frameHeight//2

    
    print(X,Y)
    cv2.drawMarker(image_out, (X,Y), (0,0,255), cv2.MARKER_STAR )
    stop = clock()
    took = stop-start
    cv2.putText(image_out,'Inference: {}s, post: {}s'.format(  np.round(inference_took,3) , np.round(took-inference_took,3) ),(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    if X != None:
        
        fra_cenx,fra_ceny=frameWidth//2,frameHeight//2
        sc_cenx,sc_ceny=screen_sizex//2,screen_sizey//2
        sc_to_frx=screen_sizex//frameWidth
        sc_to_fry=screen_sizey//frameHeight
        off_x,off_y=X - fra_cenx,Y - fra_ceny
        off_x,off_y=off_x*sc_to_frx*3*kx,off_y*sc_to_fry*3*ky
        x,y=off_x+sc_cenx,off_y+sc_ceny
        pg.moveTo(x,y,2)


    if visual:
        cv2.imshow("OpenPose's stolen hand tracking network in Keras", image_out)
        cv2.waitKey(1)

cap.release()




