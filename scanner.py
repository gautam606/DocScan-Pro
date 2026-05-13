import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < 10 or maxHeight < 10:
        return image

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def detect_document(image):
    h, w = image.shape[:2]
    scale = 800.0 / max(h, w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    rh, rw = resized.shape[:2]

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Edge detection
    edged = cv2.Canny(filtered, 30, 100)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(filtered, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    combined = cv2.bitwise_or(thresh, edged)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    min_area = rw * rh * 0.08
    doc_contour = None

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_contour = approx.reshape(4, 2).astype("float32") / scale
            break

    if doc_contour is None:
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx.reshape(4, 2).astype("float32") / scale
                break

    if doc_contour is None:
        # Smart fallback: tight crop with 2% padding
        pad_x, pad_y = int(w * 0.02), int(h * 0.02)
        doc_contour = np.array([
            [pad_x, pad_y], [w - pad_x, pad_y],
            [w - pad_x, h - pad_y], [pad_x, h - pad_y]
        ], dtype="float32")

    return doc_contour


def remove_shadow(image):
    # Split channels and normalize each to remove shadow
    result = np.zeros_like(image)
    for i in range(3):
        ch = image[:, :, i]
        dilated = cv2.dilate(ch, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(ch, bg)
        result[:, :, i] = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return result


def apply_filter(image, filter_type):
    if filter_type == "bw":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Denoise before threshold for cleaner result
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        result = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 25, 12)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    elif filter_type == "grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    elif filter_type == "high_contrast":
        shadow_removed = remove_shadow(image)
        lab = cv2.cvtColor(shadow_removed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    elif filter_type == "sharp":
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 2.0, blurred, -1.0, 0)
        return sharpened

    elif filter_type == "brightness":
        return cv2.convertScaleAbs(image, alpha=1.4, beta=50)

    elif filter_type == "original":
        return image

    return image


def auto_rotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 1:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def rotate_image(input_path, output_path, direction="right"):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Could not read image")
    if direction == "right":
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif direction == "left":
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite(output_path, rotated)


def scan_document(input_path, output_path, filter_type="bw"):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Could not read image")

    pts = detect_document(image)
    warped = four_point_transform(image, pts)

    # Remove shadow before applying filter (except original)
    if filter_type not in ("original", "brightness"):
        warped = remove_shadow(warped)

    result = apply_filter(warped, filter_type)

    # Light denoise for non-BW filters
    if filter_type not in ("bw", "original"):
        result = cv2.medianBlur(result, 3)

    cv2.imwrite(output_path, result)
    return output_path
