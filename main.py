import os
import argparse
import cv2
import numpy as np
import math
from functools import cmp_to_key
from pyzbar import pyzbar
from gen_code import *

parser = argparse.ArgumentParser(description='DIP Final Project')
# args
parser.add_argument('--img-dir', default='train', help='diretory of images')
parser.add_argument('--ticket-dir', default='tickets', help='diretory of tickets')
parser.add_argument('--qr-dir', default='qr', help='diretory of QR code')

args = parser.parse_args()


ticket_w = 1200
ticket_h = 720
code_w_b = int(ticket_w / 10 * 3)
code_w_e = 5
code_h_b = int(ticket_h / 15 * 7)
code_h_e = 50
code_w = 407
anchor_e = int(code_w / 4)

def get_points(img, bound=20):
	points = None
	# 霍夫变换找直线
	edges = cv2.Canny(img, 50, 150, 3)
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=80, maxLineGap=50)
	# 在有足够边、交点的情况下，用边的交点做透视变换，否则用外接矩形做透视变换
	if lines is not None:
		# 去除较为接近的边
		line_arg = []
		for i in range(len(lines)):
			x1, y1, x2, y2 = lines[i, 0]
			flag = True
			for j in range(len(line_arg)):
				k, b = line_arg[j]
				if k == 1e10 and abs(x1 - b) + abs(x2 - b) < bound:
					flag = False
					break
				if abs((k * x1 + b - y1) / math.sqrt(1 + k ** 2)) + abs(
						(k * x2 + b - y2) / math.sqrt(1 + k ** 2)) < bound:
					flag = False
					break
			if not flag:
				continue
			k = 1e10 if x1 == x2 else (y1 - y2) / (x1 - x2)
			b = x1 if x1 == x2 else y2 - x2 * k
			line_arg.append([k, b])
		line_arg.sort(key=lambda x: (abs(x[0]) if x[0] < -3 else x[0]))
		# 有四条直线的情况下找交点
		if len(line_arg) == 4:
			points = []
			for i in range(2):
				k, b = line_arg[i]
				for j in range(2, 4):
					k1, b1 = line_arg[j]
					x = b1 if k1 == 1e10 else (b1 - b) / (k - k1)
					y = k * x + b
					points.append([x, y])
			points = np.int0(points)
	return points


def clockwise_sort(points):
	points = points[np.argsort(points[:, 0] + points[:, 1])]
	p0 = points[0]

	def crossprod_cmp(p1, p2):
		v1 = p1 - p0
		v2 = p2 - p0
		return v1[1] * v2[0] - v1[0] * v2[1]

	points[1:] = np.array(sorted(points[1:], key=cmp_to_key(crossprod_cmp)), dtype=np.uint32)
	return points


def max_contours(thresh,minArea = True):
	# 找到面积最大的轮廓
	contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	areas = []
	for i in range(len(contours)):
		areas.append(cv2.contourArea(contours[i]))
	max_id = areas.index(max(areas))
	cnt = contours[max_id]
	if minArea:
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		return box
	else:
		rect = cv2.boundingRect(cnt)
		return rect
	return None


def crop_ticket(img):
	# 找到轮廓
	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	box = max_contours(thresh)
	mask = np.zeros(thresh.shape, dtype=np.uint8)
	cv2.fillPoly(mask, [box], (255))
	thresh = cv2.add(thresh, np.zeros(thresh.shape, dtype=np.uint8), mask=mask)
	# 闭操作去除无用纹路
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
	points = get_points(thresh)
	if points is not None:
		box = points
	# 将顶点从票面左上角顺时针排序
	box = clockwise_sort(box)
	if np.linalg.norm(box[1] - box[0]) < np.linalg.norm(box[2] - box[1]):
		box = np.concatenate((box[1:], box[:1]))
	# 透视变换
	box = box.astype(np.float32)
	target = np.array([[0, 0], [ticket_w, 0], [ticket_w, ticket_h], [0, ticket_h]], dtype=np.float32)
	matrix = cv2.getPerspectiveTransform(box, target)
	perspective_img = cv2.warpPerspective(img, matrix, (ticket_w, ticket_h))
	# 根据二维码位置判断是否将票面旋转180度
	imgray = cv2.cvtColor(perspective_img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 32, 255, cv2.THRESH_BINARY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	thresh = cv2.erode(thresh, kernel, iterations=4)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
	thresh = cv2.dilate(thresh, kernel, iterations=4)
	if np.sum(thresh[code_h_e:code_h_b, code_w_e:code_w_b]) < np.sum(thresh[-code_h_b:-code_h_e, -code_w_b:-code_w_e]):
		M = cv2.getRotationMatrix2D((ticket_w * 0.5, ticket_h * 0.5), 180, 1)
		perspective_img = cv2.warpAffine(perspective_img, M, (ticket_w, ticket_h))
	# cv2.imshow('',perspective_img)
	# if cv2.waitKey(0) & 0xff ==ord('q'):
	# 	cv2.imshow('',img)
	# 	cv2.waitKey(0)
	return perspective_img

def crop_qr_code(img):
	# 截出二维码的大致区域并二值化
	qr_img = img[-code_h_b:-code_h_e, -code_w_b:-code_w_e]
	imgray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	thresh1 = cv2.bitwise_not(thresh)
	# 用较小的kernel做闭运算，此时二维码连成一块
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	thresh = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=5)
	# 用较大的kernel做开运算，此时其余图案被清除
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
	box = max_contours(thresh)
	mask = np.zeros(thresh.shape, dtype=np.uint8)
	cv2.fillPoly(mask, [box], (255))
	thresh = cv2.add(thresh1, np.zeros(thresh.shape, dtype=np.uint8), mask=mask)
	# 闭操作将二维码连成整块
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=10)
	# 将原本图像外接上边框为5的边缘，避免mask接触到边界找不到边
	thresh1 = np.zeros((thresh.shape[0] + 10, thresh.shape[1] + 10), dtype=np.uint8)
	box = max_contours(thresh)
	thresh1[5:-5, 5:-5] = thresh
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=10)
	points = get_points(thresh1)
	if points is not None:
		for i in range(len(points)):
			x, y = points[i] - 4
			x = 0 if x < 0 else x
			y = 0 if y < 0 else y
			box[i] = [x, y]
	# 对四个点进行排序，并做透视变换
	box = clockwise_sort(box)
	box = box.astype(np.float32)
	target = np.array([[0, 0], [code_w, 0], [code_w, code_w], [0, code_w]], dtype=np.float32)
	matrix = cv2.getPerspectiveTransform(box, target)
	perspective_img = cv2.warpPerspective(qr_img, matrix, (code_w, code_w))
	# cv2.imshow('',perspective_img)
	# if cv2.waitKey(0) & 0xff == ord('q'):
	# 	cv2.imshow('',img)
	# 	cv2.waitKey(0)
	# 	cv2.imshow('',thresh1)
	# 	cv2.waitKey(0)
	return perspective_img

def adjustAnchor(thresh, bound):
	img_h,img_w = thresh.shape
	u,d,l,r = bound
	l=l+img_w if l<0 else l
	r=r+img_w if r<0 else r
	thresh1 = cv2.bitwise_not(thresh)
	thresh2 = np.zeros(thresh.shape, dtype=np.uint8)
	thresh2[u:d,l:r] = thresh1[u:d,l:r]
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=2)
	x,y,w,h = max_contours(thresh2, False)
	img_w += h-w
	new_thresh = cv2.resize(thresh,(img_w,img_h))
	new_thresh[:,0:l]=thresh[:,0:l]
	new_thresh[:,l:l+h]=cv2.resize(thresh[:,l:l+w],(h,img_h))
	new_thresh[:,l+h:]=thresh[:,l+w:]
	return new_thresh

def qr_recognition(img):
	text = ''
	# cv2.imshow('',thresh)
	# cv2.waitKey(0)
	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(imgray, ksize = (5,5),sigmaX = 0, sigmaY = 0)
	kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
	lapimg = cv2.filter2D(blur, -1, kernel)
	thresh = cv2.adaptiveThreshold(lapimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)
	barcodes = pyzbar.decode(thresh)
	text = barcodes[0].data.decode('utf-8') if len(barcodes)!=0 else ''
	if text!='':
		return text
	thresh1 = adjustAnchor(thresh,(0,anchor_e,0,anchor_e))
	thresh1 = adjustAnchor(thresh1,(0,anchor_e,-anchor_e,-1))
	if text == '':
		barcodes = pyzbar.decode(thresh1)
		text = barcodes[0].data.decode('utf-8') if len(barcodes)!=0 else ''
		if text!='':
			return text
	try:
		point_rows, point_columns = find_split_lines(thresh)
		if point_rows is not None:
			genimg = generate_image(thresh, point_rows, point_columns)
			barcodes = pyzbar.decode(genimg)
			text = barcodes[0].data.decode('utf-8') if len(barcodes)!=0 else ''
			if text !='':
				return text
		point_rows, point_columns = find_split_lines(thresh1)
		if point_rows is not None:
			genimg = generate_image(thresh1, point_rows, point_columns)
			barcodes = pyzbar.decode(genimg)
			text = barcodes[0].data.decode('utf-8') if len(barcodes)!=0 else ''
			if text !='':
				return text
	finally:
		return text
	return text

def main(img_dir, ticket_dir, qr_dir):
	total = 0
	right = 0
	if not os.path.exists(ticket_dir):
	    os.mkdir(ticket_dir)
	if not os.path.exists(qr_dir):
	    os.mkdir(qr_dir)
	f = open('prediction.txt', 'w')

	for img_name in os.listdir(img_dir):
	# if True:
		# img_name = "0010.bmp"
		img_path = os.path.join(img_dir, img_name)
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)

		ticket = crop_ticket(img)
		qr_code = crop_qr_code(ticket)
		result = qr_recognition(qr_code)

		total += 1
		if result != '':
			right += 1

		print(img_name + " " + result)

		ticket_path = os.path.join(ticket_dir, img_name)
		qr_path = os.path.join(qr_dir, img_name)
		cv2.imwrite(ticket_path, ticket)
		cv2.imwrite(qr_path, qr_code)
		f.write(img_name + " " + result + "\n")

		# break

	f.close()
	print("{:.2f}%".format(right / total * 100))


if __name__ == "__main__":
	main(args.img_dir, args.ticket_dir, args.qr_dir)