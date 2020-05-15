import cv2
import numpy as np
from tqdm import tqdm
from pytesseract import image_to_string

def thresher(img, const = 13):
    # colour
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur
    temp = cv2.GaussianBlur(temp, (5,5), 0)
    # adaptive thresh
    temp = cv2.adaptiveThreshold(temp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,9,const)
    return temp

def thrimg_text_cls(thr_img,min_thresh = 200, max_thresh = 2000,min_cc_number = 5):
    # finding connected components
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thr_img)
    
    # filtering out small ones
    valid_idx = np.where(np.logical_and(stats[:,4] >= min_thresh,stats[:,4] <= max_thresh))
        
    # classifying using threshold
        # threshold is 1/10 of image height
    dispersion_thr = thr_img.shape[0]/10
    return np.std(centroids[valid_idx][:,1]) < dispersion_thr and len(valid_idx[0]) >= min_cc_number

def tesseract_recognition(video_path):
    # Checking classification
    ## Minimum time for subs is 0.5 seconds
    
    cap= cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: could not open video"
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # beginning position
    #cap.set(1,2*60*fps)
    ret, frame = cap.read()
    if ret == False:
        return "Error: error while reading frame"
        
    ans = []
    old_thr = thresher(frame)
    previous_clean = np.zeros((1080,1920),np.uint8)
    for frame_num in tqdm(range(1,total_frames)):
        #reading
        ret, frame = cap.read()
        if ret == False:
            return "Error: error while reading frame"
        # finding contours
        thr = thresher(frame)
        # logical and for finding ststic zones with previous frame
        thr_and = thr&old_thr
        old_thr = thr
        # filling holes
        kernel = np.array([[1,1,1],[0,0,0],[1,1,1]],np.uint8)
        dil = cv2.dilate(thr_and,kernel,iterations = 1)
        # finding black pixel for flooding
        zero_pos = (dil==0).nonzero()
        zero_pos = (zero_pos[0][0],zero_pos[1][0])
        
        # flooding image
        flood_img = np.copy(dil)
        mask = np.zeros((H+2, W+2), np.uint8)
        cv2.floodFill(flood_img,mask,zero_pos,255)
        # inverting
        clean = cv2.bitwise_not(flood_img)
        clf = thrimg_text_cls(clean)
        if clf:
            if np.count_nonzero(clean & previous_clean) < 0.6 * np.count_nonzero(clean):
                ans.append(image_to_string(flood_img,lang='rus'))
                previous_clean = clean
        else:
            previous_clean = np.zeros((1080,1920),np.uint8)
    
    cap.release()
    cv2.destroyAllWindows()
    return ans