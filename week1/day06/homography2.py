import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest diff
    return rect

def auto_perspective_correct(img_path, output_size=(600, 850)):
    img = cv2.imread(img_path)
    orig = img.copy()
    
    # Step 1: Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    brads = cv2.imshow("edges", edges)
   
    # Step 2: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # # Step 1: Preprocess
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # # Lower the Canny thresholds slightly to catch softer edges
    # edges = cv2.Canny(blurred, 50, 150) 
    
    # # FIX: Add Dilation to close gaps in the edge map
    # kernel = np.ones((5, 5), np.uint8)
    # edges = cv2.dilate(edges, kernel, iterations=2) # Thickens the edges
    # brads = cv2.imshow("edges", edges)
    
    # # Step 2: Find contours
    # # Using RETR_EXTERNAL ignores contours inside other contours, 
    # # which helps prevent it from snapping to images inside the document.
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Step 3: Find the 4-corner contour (the document)
    doc_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_contour = approx
            break
    
    if doc_contour is None:
        print("Could not find document automatically — use manual version")
        return None
    
    # Step 4: Order the points and warp
    pts = doc_contour.reshape(4, 2).astype(np.float32)
    src = order_points(pts)
    
    w, h = output_size
    dst = np.float32([[0,0], [w,0], [w,h], [0,h]])
    
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (w, h))
    
    # Visualize
    cv2.drawContours(orig, [doc_contour], -1, (0, 255, 0), 3)
    for pt in src:
        cv2.circle(orig, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[1].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Detected corners')
    axes[2].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Perspective corrected')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig('output/auto_correction.jpg', dpi=150)
    plt.show()
    plt.show(brads)
    return warped

# result = auto_perspective_correct('data/document_angled.jpg')
result = auto_perspective_correct('week1/day06/distort_2.png')
