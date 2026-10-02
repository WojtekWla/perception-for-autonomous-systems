import cv2 as cv
import glob


images = sorted(glob.glob("weekly_project_5/car_challange/rgb/*.jpg"))
depth = sorted(glob.glob("weekly_project_5/car_challange/depth/*.png"))

print(len(images), len(depth))
i = 1
for img, d in zip(images, depth):
    print(f"Image number {i}")
    img = cv.imread(img)
    depth = cv.imread(d)

    cv.imshow("image", img)
    cv.imshow("depth", depth)
    if cv.waitKey(0) == 27:
        break
    i += 1

cv.destroyAllWindows()
