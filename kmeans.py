import cv2
import numpy as np

# Function to display image
def view(win_name, img):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    resize_factor = 1.8
    h, w = img.shape[:2]
    h = int(h / resize_factor)
    w = int(w / resize_factor)
    cv2.resizeWindow(win_name, w, h)
    cv2.imshow(win_name, img)

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img, gray, thresh

# Function to extract features using contours
def extract_features(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to perform clustering on contours
def cluster_contours(contours, img):
    mask = np.zeros(img.shape[:2], np.uint8)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    Z = mask.reshape((-1, 1))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 2
    _, labels, _ = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    clustered = labels.reshape((mask.shape))
    return clustered

# Function to highlight handwritten parts
def highlight_handwritten_parts(clustered, img):
    handwritten_mask = (clustered == 1).astype(np.uint8) * 255

    contours, _ = cv2.findContours(handwritten_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cd = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cd.append([x, y])
        cd.append([x + w, y + h])
    return img, cd

# Main function
if __name__ == "__main__":
    name = "861.tif"
    image_path = fr"C:\Users\KIIT\Downloads\ 1 - Copy (2){name}"
    img, gray, thresh = preprocess_image(image_path)
    contours = extract_features(thresh)
    clustered = cluster_contours(contours, img)
    result_img, cd = highlight_handwritten_parts(clustered, img)

    view('Detected Handwritten Parts', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    height, width = img.shape[:2]
    target = np.zeros((height, width, 3), np.uint8)
    for i in range(0, len(cd), 2):
        cv2.rectangle(target, (cd[i][0], cd[i][1]), (cd[i + 1][0], cd[i + 1][1]), (255, 255, 255), -1)
    view('Resulting Mask', target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(r"C:\Users\KIIT\Downloads\ 1 - Copy (2)/{name}", target)
