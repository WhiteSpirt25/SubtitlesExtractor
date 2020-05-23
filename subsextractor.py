import cv2
import numpy as np
from tqdm import tqdm
from pytesseract import image_to_string

class subtitle_extractor:  
    def __init__(self):
        self.text = None
        self.time = None
    
    def thresher(self,img, const = 13):
        # colour
        temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur
        temp = cv2.GaussianBlur(temp, (5,5), 0)
        # adaptive thresh
        temp = cv2.adaptiveThreshold(temp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,9,const)
        return temp

<<<<<<< HEAD
    def thrimg_text_cls(self,thr_img,min_thresh = 200, max_thresh = 2000,min_cc_number = 5):
        # finding connected components
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thr_img)

        # filtering out small ones
        valid_idx = np.where(np.logical_and(stats[:,4] >= min_thresh,stats[:,4] <= max_thresh))

        # classifying using threshold
            # threshold is 1/10 of image height
        dispersion_thr = thr_img.shape[0]/10
        return np.std(centroids[valid_idx][:,1]) < dispersion_thr and len(valid_idx[0]) >= min_cc_number

    def timestamp(self,frame_num,fps):
        s = frame_num / fps
        s,ms = divmod(s, 1)
        ms = int(ms * 1000) # because ms was a procent of a second
        m,s = divmod(s,60)
        h,m = divmod(m,60)
        return "{:02d}:{:02d}:{:02d},{:03d}".format(int(h),int(m),int(s),int(ms))

    def tesseract_recognition(self,video_path, lang = 'eng',percent_to_recog = 1.0):
        # Checking classification
        ## Minimum time for subs is 0.5 seconds

        cap= cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: could not open video"
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * percent_to_recog)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # beginning position
        #cap.set(1,2*60*fps)
        ret, frame = cap.read()
        if ret == False:
            return "Error: error while reading frame"

        text = []
        time = []
        part_with_text = False # used for correct timing
        old_thr = self.thresher(frame)
        previous_clean = np.zeros((1080,1920),np.uint8)
        for frame_num in tqdm(range(1,total_frames)):
            #reading
            ret, frame = cap.read()
            if ret == False:
                return "Error: error while reading frame"
            # finding contours
            thr = self.thresher(frame)
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
            # checking for text
            clf = self.thrimg_text_cls(clean,min_thresh=200,max_thresh=3000,min_cc_number=5)
            if clf:
                # comparing with previous frame to check if different and needs to be saved
                if np.count_nonzero(clean & previous_clean) < 0.6 * np.count_nonzero(clean):
                    text.append(image_to_string(flood_img,lang=lang))
                    previous_clean = clean
                    if part_with_text:
                        time.append(self.timestamp(frame_num,fps))
                        time.append(self.timestamp(frame_num,fps)) #because it's beginning of new subtitle line
                    else:
                        time.append(self.timestamp(frame_num,fps))
                        part_with_text = True

            else:
                # ending text part and refreshing previous text frame
                previous_clean = np.zeros((1080,1920),np.uint8)
                if part_with_text:
                    time.append(self.timestamp(frame_num,fps))
                    part_with_text = False

        cap.release()
        cv2.destroyAllWindows()

        # changing time to be in pairs 
        if len(time)%2 ==0:
            time_paired = [(time[i],time[i+1]) for i in range(0,len(time),2)]
        else:
            time_paired = [(time[i],time[i+1]) for i in range(0,len(time)-1,2)]
            
        self.text = text
        self.time = time_paired

    def save_srt(self,name):
        # checking data
        if len(self.text) != len(self.time):
            f = open(f"{name}.srt", "w")    
            for i in range(len(self.time)):
                f.write(str(i+1)+'\n')
                f.write(f"{self.time[i][0]} --> {self.time[i][1]}\n")
                f.write(self.text[i]+'\n')
                f.write('\n')
            f.close()
        # creating file and writing data to it in srt format
        f = open(f"{name}.srt", "w")    
        for i in range(len(self.text)):
            f.write(str(i+1)+'\n')
            f.write(f"{self.time[i][0]} --> {self.time[i][1]}\n")
            f.write(self.text[i]+'\n')
            f.write('\n')
        f.close()
=======
def timestamp(frame_num,fps):
    s = frame_num / fps
    s,ms = divmod(s, 1)
    ms = int(ms * 1000) # because ms was a procent of a second
    m,s = divmod(s,60)
    h,m = divmod(m,60)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(int(h),int(m),int(s),int(ms))

def tesseract_recognition(video_path, lang = 'eng'):
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
        
    text = []
    time = []
    part_with_text = False # used for correct timing
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
        # checking for text
        clf = thrimg_text_cls(clean,min_thresh=Min_thresh,max_thresh=Max_thresh,min_cc_number=Min_cc_number)
        if clf:
            # comparing with previous frame to check if different and needs to be saved
            if np.count_nonzero(clean & previous_clean) < 0.6 * np.count_nonzero(clean):
                ans.append(image_to_string(flood_img,lang=lang))
                previous_clean = clean
                if part_with_text:
                    time.append(timestamp(frame_num,fps))
                    time.append(timestamp(frame_num,fps)) #because it's beginning of new subtitle line
                else:
                    time.append(timestamp(frame_num,fps))
                    part_with_text = True
                
        else:
            # ending text part and refreshing previous text frame
            previous_clean = np.zeros((1080,1920),np.uint8)
            if part_with_text:
                time.append(timestamp(frame_num,fps))
                part_with_text = False
    
    cap.release()
    cv2.destroyAllWindows()
    
    # changing time to be in pairs 
    time_paired = [(time[i],time[i+1]) for i in range(0,len(time),2)]
    return ans,time_paired

def save_srt(name,text,time):
    # checking data
    if len(text) != len(time):
        return "Error: subtitle text and time is different size"
    # creating file and writing data to it in srt format
    f = open(f"{name}.srt", "w")    
    for i in range(len(text)):
        f.write(str(i+1)+'\n')
        f.write(f"{time[i][0]} --> {time[i][1]}\n")
        f.write(text[i]+'\n')
        f.write('\n')
    f.close()
>>>>>>> e0631e76c94492fae1ac21ca98acc3ca59e07ebc
