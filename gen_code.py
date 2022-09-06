import os
import argparse
import cv2
import numpy as np
import math

def found_point(mat, rows=True):
	temp1 = []
	i = 0
	normal_gap = 5
	last_gap = 0
	length = mat.shape[0] if rows else mat.shape[1]
	while i < length:
		if i + 11 < length:
			if i < length - 20:
				gap = normal_gap
			else:
				gap = last_gap
			if rows:
				point_row = Point(mat[int(i) + gap:min(int(i) + 18, 406), :], min(12, 406 - int(i)), rows)
			else:
				point_row = Point(mat[:, int(i) + gap:min(int(i) + 18, 406)], min(12, 406 - int(i)), rows)
			mat_temp = point_row + gap if point_row is not None else None
		else:
			break
		if mat_temp is None:
			i = i + 11
		else:
			i = i + mat_temp
			temp1.append(i)
	return temp1

def Point(mat, border, rows):
	gap = 3
	result_array = []
	white_tmp_matrix = np.zeros((1, mat.shape[1]), np.int16) if rows else np.zeros((mat.shape[0], 1), np.int16)
	black_last_matrix = np.zeros((1, mat.shape[1]), np.int16) if rows else np.zeros((mat.shape[0], 1), np.int16)
	length = mat.shape[0] if rows else mat.shape[1]
	# 从白色点变为黑色点,白色连续线段超过2，下一个点为黑点
	for i in range(0, min(border, length)):
		if i == 0:
			white_tmp_matrix = np.where(mat[i, :] == 255, 1, 0) if rows else np.where(mat[:, i] == 255, 1, 0)
		else:
			white_current_matrix = np.where(mat[i, :] == 255, 1, -1) if rows else np.where(mat[:, i] == 255, 1, -1)
			white_cal_matrix = white_tmp_matrix * white_current_matrix
			white_count_matrix = np.where(white_cal_matrix == -1, 1, 0)
			white_change_count = white_count_matrix.sum()
			# 变更tmp，将已经变化的点修改为0
			white_tmp_matrix = white_tmp_matrix * np.where(white_cal_matrix == -1, 0, 1)
			# j > gap宽度则计入计算
			if i > gap and white_change_count > 0:
				for k in range(0, white_change_count):
					result_array.append(i - 1)
		if i == 0:
			black_last_matrix = np.where(mat[i, :] == 0, 1, 0) if rows else np.where(mat[:, i] == 0, 1, 0)
		else:
			black_current_matrix = np.where(mat[i, :] == 0, 5, 1) if rows else np.where(mat[:, i] == 0, 5, 1)
			black_cal_matrix = black_last_matrix + black_current_matrix
			# 统计上次为黑本次为白
			black_count_matrix = np.where(black_cal_matrix == 2, 1, 0)
			black_change_count = black_count_matrix.sum()
			# 变更tmp
			black_last_matrix = black_cal_matrix
			# 上次为白，本次为白，更新为0
			black_last_matrix = np.where(black_last_matrix == 1, 0, black_last_matrix)
			# 上次为黑，本次为白，更新为1000
			black_last_matrix = np.where(black_last_matrix == 2, 1000, black_last_matrix)
			# 上次为白，本次为黑，更新为1
			black_last_matrix = np.where(black_last_matrix == 5, 1, black_last_matrix)
			# 上次为黑，本次为黑，更新为1
			black_last_matrix = np.where(black_last_matrix == 6, 1, black_last_matrix)
			# j > gap宽度则计入计算
			if i > gap and black_change_count > 0:
				for k in range(0, black_change_count):
					result_array.append(i - 1)
	if len(result_array)==0:
		return None
	result = max(int(np.median(np.array(result_array))), int((np.array(result_array).mean())))
	return result

# 找到图片的XY分割线
def find_split_lines(image):
	point_rows = found_point(image)
	point_columns = found_point(image, False)
	point_columns = point_columns[0:36] if len(point_columns)>36 else point_columns
	point_rows = point_rows[0:36] if len(point_rows)>36 else point_rows
	# draw_line_image = np.copy(image)
	# for rowIndex in range(0, len(point_rows)):
	# 	cv2.line(draw_line_image, (0, point_rows[rowIndex]), (406, point_rows[rowIndex]), (0, 0, 0), 1)
	# for colIndex in range(0, len(point_columns)):
	# 	cv2.line(draw_line_image, (point_columns[colIndex], 0), (point_columns[colIndex], 406), (0, 0, 0), 1)
	# cv2.imshow('',draw_line_image)
	# cv2.waitKey(0)
	if len(point_columns) != 36 or len(point_rows) != 36:
		return None, None
	# 增补中止线
	point_rows.append(407)
	point_columns.append(407)
	return point_rows, point_columns

def generate_image(image, point_rows, point_columns):
	margin_offset = 0
	round_offset = 1
	gen_img = np.ones((407, 407), np.uint8)*255
	gen_img[0:407, 0:407] = 255
	for i in range(-1, len(point_rows) - 1):
		margin_offset = 0
		for j in range(-1, len(point_columns) - 1):
			start_x = 0 if j == -1 else point_columns[j]
			start_y = 0 if i == -1 else point_rows[i]
			end_x = point_columns[j + 1]
			end_y = point_rows[i + 1]
			avg_center = image[start_y + round_offset:end_y - round_offset,start_x + round_offset:end_x - round_offset]
			avg_total = image[start_y:end_y, start_x:end_x]
			avg_offset_3_center = image[start_y + round_offset:end_y - round_offset, start_x:start_x + 3]
			# 整体偏黑判定为黑色
			avg_center_mean = avg_center.mean()
			avg_offset_3_center_mean = avg_offset_3_center.mean()
			avg_inner_round_mean = (avg_total.sum() - avg_center.sum()) / (
					avg_total.shape[0] * avg_total.shape[1] - avg_center.shape[0] * avg_center.shape[1])
			# 如果整体判定为黑色，判断是否存在向右偏移的情况
			# 当前格子的偏移将影响下一个格子的偏移情况
			if avg_center_mean < 50:
				current_offset = int(round(avg_offset_3_center_mean / 255.0 * 3))
				margin_offset = current_offset if margin_offset < current_offset else margin_offset
				if margin_offset == None:
					margin_offset = 0
			if avg_center_mean < 50:
				gen_img[i * 11 + 11:i * 11 + 22, j * 11 + 11: j * 11 + 22] = 0  # 表示为黑色
			elif avg_center_mean > 150:
				# 中心点偏白判定为白色
				gen_img[i * 11 + 11:i * 11 + 22, j * 11 + 11: j * 11 + 22] = 255  # 表示为白色
			elif avg_center_mean > avg_inner_round_mean:
				gen_img[i * 11 + 11:i * 11 + 22, j * 11 + 11: j * 11 + 22] = 255  # 表示为白色
			else:
				gen_img[i * 11 + 11:i * 11 + 22, j * 11 + 11: j * 11 + 22] = 0
	gen_img = fill_detect(gen_img)
	return gen_img

# 根据先验知识绘制定位点和校验位
def fill_detect(image):
	image[0:77, 0:77] = 0# 重构左上定位点
	image[11:66, 11:66] = 255
	image[22:55, 22:55] = 0
	image[330:407, 0:77] = 0# 重构左下定位点
	image[341:396, 11:66] = 255
	image[352:385, 22:55] = 0
	image[0:77, 330:407] = 0# 重构右上定位点
	image[11:66, 341:396] = 255
	image[22:55, 352:385] = 0
	image[66:77, 77:330] = 255# 重构横向校验位
	image[66:77, 88:99] = 0
	image[66:77, 110:121] = 0
	image[66:77, 132:143] = 0
	image[66:77, 154:165] = 0
	image[66:77, 176:187] = 0
	image[66:77, 198:209] = 0
	image[66:77, 220:231] = 0
	image[66:77, 242:253] = 0
	image[66:77, 264:275] = 0
	image[66:77, 286:297] = 0
	image[66:77, 308:319] = 0
	image[77:330, 66:77] = 255# 重构纵向校验位
	image[88:99, 66:77] = 0
	image[110:121, 66:77] = 0
	image[132:143, 66:77] = 0
	image[154:165, 66:77] = 0
	image[176:187, 66:77] = 0
	image[198:209, 66:77] = 0
	image[220:231, 66:77] = 0
	image[242:253, 66:77] = 0
	image[264:275, 66:77] = 0
	image[286:297, 66:77] = 0
	image[308:319, 66:77] = 0
	image[308:363, 308:363] = 0# 重构右下小定位点
	image[319:352, 319:352] = 255
	image[330:341, 330:341] = 0
	return image