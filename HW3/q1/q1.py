

import cv2
from function import *


def main(name):
    img = cv2.imread(name)
    dim = (img.shape[1], img.shape[0])
    img = cv2.resize(img, (min(dim), min(dim)))
    
    [acc, edge] = hough_space(img)

    lines = find_lines(img, acc, thresh=200)
    lines = avg_near_line(lines)
    chess_lines = find_chess_lines(img, lines)
    chess_lines = select_parallel_lines(chess_lines, 2)
    corners = intersection(img, chess_lines)

    img_line = cv2.resize(draw_line(img, make_mb(lines)), dim)
    img_chess_line = cv2.resize(draw_line(img, make_mb(chess_lines)), dim)

    img_corners = img.copy()
    for row in corners:
        img_corners = cv2.circle(img_corners, (row[0], row[1]), 2, (0, 255, 0), 4)
    img_corners = cv2.resize(img_corners, dim)

    edge = cv2.resize(edge, dim)
    acc_big = cv2.resize(acc, (int(acc.shape[1] * 3), acc.shape[0] // 2))
    acc_big = normalize(3 * normalize(acc_big))
    return [edge, acc, acc_big, img_line, img_chess_line, img_corners]


[edge1, acc1, acc_big1, img_line1, img_chess_line1,img_corners1] = main('im01.jpg')
cv2.imwrite('res01.jpg', edge1)
cv2.imwrite('res03-hough-space.jpg', acc1)
cv2.imwrite('res03-hough-space-big.jpg', acc_big1)
cv2.imwrite('res05-lines.jpg', img_line1)
cv2.imwrite('res07-chess.jpg', img_chess_line1)
cv2.imwrite('res09-corners.jpg', img_corners1)

[edge2, acc2, acc_big2, img_line2, img_chess_line2,img_corners2] = main('im02.jpg')
cv2.imwrite('res02.jpg', edge2)
cv2.imwrite('res04-hough-space.jpg', acc2)
cv2.imwrite('res04-hough-space-big.jpg', acc_big2)
cv2.imwrite('res06-lines.jpg', img_line2)
cv2.imwrite('res08-chess.jpg', img_chess_line2)
cv2.imwrite('res10-corners.jpg', img_corners2)
