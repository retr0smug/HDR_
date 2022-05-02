import numpy as np
from scipy.misc.pilutil import imresize
import cv2, sys, time
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


DIGIT_WIDTH = 10
DIGIT_HEIGHT = 20
IMG_HEIGHT = 28
IMG_WIDTH = 28
CLASS_N = 10 # 0-9

##-------------------------------------------------------------------------------------------------##
#                  IMAGE PREPARATION                 #
##-------------------------------------------------------------------------------------------------##
def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_digits(fn):
    print('Loading "%s for training"...' % fn)
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (DIGIT_WIDTH, DIGIT_HEIGHT))
    resized_digits = []
    for digit in digits:
        resized_digits.append(imresize(digit,(IMG_WIDTH, IMG_HEIGHT)))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return np.array(resized_digits), labels

def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img, 
                 orientations=10, 
                 pixels_per_cell=(5,5),
                 cells_per_block=(1,1), 
                 visualize=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)

##-------------------------------------------------------------------------------------------------##
#                 CUSTOM CLASSIFIERS                 #
##-------------------------------------------------------------------------------------------------##

#######---KNN 
class KNN_MODEL():
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()
#######---SVM
class SVM_MODEL():
    def __init__(self, num_feats, C = 1, gamma = 0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF) #SVM_LINEAR, SVM_RBF
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.features = num_feats

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        results = self.model.predict(samples.reshape(-1,self.features))
        return results[1].ravel()

##-------------------------------------------------------------------------------------------------##
#                 LOADING IMAGE                      #
##-------------------------------------------------------------------------------------------------##

def get_digits(contours, hierarchy):
    hierarchy = hierarchy[0]
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]   
    final_bounding_rectangles = []
    #find the most common heirarchy level - that is where our digits's bounding boxes are
    u, indices = np.unique(hierarchy[:,-1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]
    
    for r,hr in zip(bounding_rectangles, hierarchy):
        x,y,w,h = r
        #this could vary depending on the image you are trying to predict
        #we are trying to extract ONLY the rectangles with images in it (this is a very simple way to do it)
        #we use heirarchy to extract only the boxes that are in the same global level - to avoid digits inside other digits
        #ex: there could be a bounding box inside every 6,9,8 because of the loops in the number's appearence - we don't want that.
        if ((w*h)>250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy: 
            final_bounding_rectangles.append(r)    

    return final_bounding_rectangles

def blank_IMG(img_file):
    #Returns True if all white pixels or False if not all white
    img = cv2.imread(img_file)
    pixel_mean = np.mean(img)
    #print(pixel_mean)
    if np.ceil(pixel_mean) >= 250:
        return 1
    """H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    pixels = cv2.countNonZero(thresh)
    return True if pixels == (H * W) else False"""

def proc_user_img(img_file, model):
    print('Loading "%s for digit recognition"...' % img_file)
    im = cv2.imread(img_file)    
    blank_image = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
    blank_image.fill(255)

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    kernel = np.ones((5,5),np.uint8)
        
    ret,thresh = cv2.threshold(imgray,127,255,0)   
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
        
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    digits_rectangles = get_digits(contours,hierarchy)  #rectangles of bounding the digits in user image
        
    for rect in digits_rectangles:
        x,y,w,h = rect
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        im_digit = imgray[y:y+h,x:x+w]
        im_digit = (255-im_digit)
        im_digit = imresize(im_digit,(IMG_WIDTH ,IMG_HEIGHT))

        hog_img_data = pixels_to_hog_20([im_digit])  
        pred = model.predict(hog_img_data)
        cv2.putText(im, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        cv2.putText(blank_image, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

    plt.imshow(im)
    cv2.imwrite("original_overlay.png",im) 
    cv2.imwrite("predicted_digits.png",blank_image) 
    #cv2.destroyAllWindows()           


def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]  #row-wise ordering


#this function processes a custom training image
#see example : custom_train.digits.jpg
#if you want to use your own, it should be in a similar format
def load_digits_custom(img_file):
    train_data = []
    train_target = []
    start_class = 1
    im = cv2.imread(img_file)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    plt.imshow(imgray)
    kernel = np.ones((5,5),np.uint8)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)   
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    digits_rectangles = get_digits(contours,hierarchy)  #rectangles of bounding the digits in user image
    
    #sort rectangles accoring to x,y pos so that we can label them
    digits_rectangles.sort(key=lambda x:get_contour_precedence(x, im.shape[1]))
    
    for index,rect in enumerate(digits_rectangles):
        x,y,w,h = rect
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        im_digit = imgray[y:y+h,x:x+w]
        im_digit = (255-im_digit)
        
        im_digit = imresize(im_digit,(IMG_WIDTH, IMG_HEIGHT))
        train_data.append(im_digit)
        train_target.append(start_class%10)

        if index>0 and (index+1) % 10 == 0:
            start_class += 1
    cv2.imwrite("training_box_overlay.png",im)
    
    return np.array(train_data), np.array(train_target)

##-------------------------------------------------------------------------------------------------##
#                   DATA PREPARATION                 #
##-------------------------------------------------------------------------------------------------##

TRAIN_MNIST_IMG = 'digits.png' 
TRAIN_USER_IMG = 'custom_train_digits.jpg'
TEST_USER_IMG = 'test_img3.jpg'

digits, labels = load_digits(TRAIN_MNIST_IMG)
#digits, labels = load_digits_custom(TRAIN_USER_IMG)

print('train data shape',digits.shape)
print('test data shape',labels.shape)

digits, labels = shuffle(digits, labels, random_state=256)
train_digits_data = pixels_to_hog_20(digits)
X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=0.33, random_state=42)

##-------------------------------------------------------------------------------------------------##
#              DIGIT RECOGNITION TESTING             #
##-------------------------------------------------------------------------------------------------##

while(True):
    print("""
        Choose one of the following classifiers for prediction:
        1> k Nearest Neighbours.
        2> Support Vector Machine.
        3> Exit.
        """)
    ch = input("ENTER: ").lower()
    if(ch == '1' or ch == 'knn'):
        model = KNN_MODEL(k = 3)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)

        model = KNN_MODEL(k = 3)
        model.train(train_digits_data, labels)
        if(blank_IMG(TEST_USER_IMG) == 1):
            print("Input image is blank")
        else:    
            proc_user_img(TEST_USER_IMG, model)

            print("EVALUATION ON TESTING DATA")
            print(classification_report(y_test, predictions))        
            print("\nCONFUSION MATRIX:")
            print(confusion_matrix(y_test, predictions)) 

            accuracy_val = accuracy_score(y_test, predictions)
            text = "CALCULATING ACCURACY..."
            for ch in text:
                sys.stdout.write(ch)
                sys.stdout.flush()
                time.sleep(0.5)   
            print("\nAccuracy calculated: {:.2f}".format(accuracy_val * 100),"%") 

        

    elif(ch == '2' or ch == 'svm'):
        model = SVM_MODEL(num_feats = train_digits_data.shape[1])
        model.train(X_train, y_train)
        predictions = model.predict(X_test)

        model = SVM_MODEL(num_feats = train_digits_data.shape[1])
        model.train(train_digits_data, labels)
        if(blank_IMG(TEST_USER_IMG) == 1):
            print("Input image is blank")

        proc_user_img(TEST_USER_IMG, model)

        print("EVALUATION ON TESTING DATA")
        print(classification_report(y_test, predictions))       
        print("\nCONFUSION MATRIX:")
        print(confusion_matrix(y_test, predictions))

        accuracy_val = accuracy_score(y_test, predictions) 
        text = "CALCULATING ACCURACY..."
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(0.5)   
        print("\nAccuracy calculated: {:.2f}".format(accuracy_val * 100),"%")             

    elif(ch == '3' or ch == 'exit' or ch == 'quit'):
        sys.exit(0)    
    else:
        print("INVALID CHOICE!!")
      
