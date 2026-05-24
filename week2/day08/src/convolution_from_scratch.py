import numpy as np
import matplotlib.pyplot as plt
import cv2

# =============================================
# PART 1: Implement convolution manually
# This is EXACTLY what cv2.filter2D and nn.Conv2d do internally
# =============================================

def manual_convolve2d(image, kernel):
    """
    Slide a kernel over every position in the image.
    At each position: multiply element-wise, sum the result.
    That sum becomes the output pixel value.
    """
    img_h, img_w = image.shape
    k_h, k_w     = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Pad image so output is same size as input
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)
    
    for y in range(img_h):
        for x in range(img_w):
            # Extract the local patch under the kernel
            patch = padded[y:y+k_h, x:x+k_w]
            # Element-wise multiply + sum = one output pixel
            output[y, x] = np.sum(patch * kernel)
    
    return output

# Load a test image
img = cv2.imread('data/test.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    # Create a synthetic test image if no image available
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 180
    img[80:120, 80:120] = 255
    cv2.rectangle(img, (30, 30), (170, 170), 200, 3)
img = cv2.resize(img, (128, 128)).astype(np.float32) / 255.0

# =============================================
# PART 2: Different kernels = different features
# This is exactly what a CNN learns automatically
# =============================================

kernels = {
    'Horizontal edges\n(Sobel X)': np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32),
    
    'Vertical edges\n(Sobel Y)': np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]], dtype=np.float32),
    
    'Blur / smoothing\n(Gaussian-like)': np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]], dtype=np.float32),
    
    'Sharpening': np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]], dtype=np.float32),
    
    'Diagonal edges\n(45 degrees)': np.array([
        [-1, -1,  2],
        [-1,  2, -1],
        [ 2, -1, -1]], dtype=np.float32),
    
    'Dot detector\n(Laplacian)': np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]], dtype=np.float32),
}

fig, axes = plt.subplots(2, len(kernels)+1, figsize=(20, 8))
fig.suptitle('Key Insight: Different kernels detect different features\n'
             'A CNN learns these kernels automatically from data', 
             fontsize=13, fontweight='bold')

# Show original
axes[0][0].imshow(img, cmap='gray', vmin=0, vmax=1)
axes[0][0].set_title('Original image', fontweight='bold')
axes[0][0].axis('off')
axes[1][0].axis('off')

for idx, (name, kernel) in enumerate(kernels.items()):
    # Apply the kernel manually (your implementation)
    output = manual_convolve2d(img, kernel)
    output_clipped = np.clip(output, 0, 1)
    
    # Show kernel values as a heatmap
    axes[0][idx+1].imshow(kernel, cmap='RdBu_r', 
                           vmin=-kernel.max(), vmax=kernel.max())
    axes[0][idx+1].set_title(f'Kernel\n{name}', fontsize=9)
    for i in range(3):
        for j in range(3):
            axes[0][idx+1].text(j, i, f'{kernel[i,j]:.2f}', 
                                ha='center', va='center', fontsize=7)
    
    # Show the output feature map
    axes[1][idx+1].imshow(output_clipped, cmap='gray')
    axes[1][idx+1].set_title('Feature map output', fontsize=9)
    axes[1][idx+1].axis('off')

plt.tight_layout()
plt.savefig('output/01_kernels_and_feature_maps.jpg', dpi=150)
plt.show()

print("\n=== Key insight to remember ===")
print("A CNN with 32 filters in its first layer learns 32 different kernels.")
print("Each kernel becomes a specialized feature detector.")
print("The network decides what to detect — you don't design it.")