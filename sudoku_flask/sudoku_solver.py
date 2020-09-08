# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 13:11:34 2020

@author: ashva
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 13:08:36 2020

@author: ashva
"""

from flask import Flask,render_template, request
import os
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
from werkzeug.utils import secure_filename
import numpy as np
import imutils
import cv2

app = Flask(__name__) 

MODEL_PATH = 'models/digit_classifier.h5'
model = load_model(MODEL_PATH)


def find_puzzle(image):
	# convert the image to grayscale and blur it slightly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    
    # apply adaptive thresholding and then invert the threshold map
	thresh = cv2.adaptiveThreshold(blurred, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	thresh = cv2.bitwise_not(thresh)
	
    
    # find contours in the thresholded image and sort them by size in
	# descending order
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	# initialize a contour that corresponds to the puzzle outline
	puzzleCnt = None
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# if our approximated contour has four points, then we can
		# assume we have found the outline of the puzzle
		if len(approx) == 4:
			puzzleCnt = approx
			break
    
    # apply a four point perspective transform to both the original
	# image and grayscale image to obtain a top-down bird's eye view
	# of the puzzle
	puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
	warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
	# check to see if we are visualizing the perspective transform
	
	# return a 2-tuple of puzzle in both RGB and grayscale
	return (puzzle, warped)

def extract_digit(cell):
    thresh = cv2.threshold(cell, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts)==0:
        return None
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    
    if percentFilled < 0.03:
        return None
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    print("SUCCESSS")
    return digit


@app.route('/') 
def home(): 
    return render_template('home.html')

@app.route('/result', methods = ['GET', 'POST'])
def result():
    if request.method == 'POST':
        f = request.files['photo']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
    image = cv2.imread(file_path)
    image = imutils.resize(image, width=600)
    
    (puzzled,warped) = find_puzzle(image)
    board = np.zeros((9,9), dtype="int")
    stepX = warped.shape[1]//9
    stepY = warped.shape[0]//9
    
    cellLocs = []
    for y in range(0, 9):
	# initialize the current list of cell locations
    	row = []
    	for x in range(0, 9):
    		# compute the starting and ending (x, y)-coordinates of the
    		# current cell
    		startX = x * stepX
    		startY = y * stepY
    		endX = (x + 1) * stepX
    		endY = (y + 1) * stepY
    		# add the (x, y)-coordinates to our cell locations list
    		row.append((startX, startY, endX, endY))
            # crop the cell from the warped transform image and then
    		# extract the digit from the cell
    		cell = warped[startY:endY, startX:endX]
    		digit = extract_digit(cell)
    		# verify that the digit is not empty
    		if digit is not None:
    			# resize the cell to 28x28 pixels and then prepare the
    			# cell for classification
    			roi = cv2.resize(digit, (28, 28))
    			roi = roi.astype("float") / 255.0
    			roi = img_to_array(roi)
    			roi = np.expand_dims(roi, axis=0)
    			pred = model.predict(roi).argmax(axis=1)[0]
    		board[y, x] = pred
	# add the row to our cell locationss
    	cellLocs.append(row)
    
    puzzle = Sudoku(3, 3, board=board.tolist())
    puzzle.show()
    # solve the Sudoku puzzle
    solution = puzzle.solve()
    solution.show_full()
    answer = str(solution).split("\n")
    return render_template('result.html', Solution = answer[5:])
    
# main driver function 
if __name__ == '__main__': 
  
    # run() method of Flask class runs the application  
    # on the local development server. 
    app.run(debug=True) 