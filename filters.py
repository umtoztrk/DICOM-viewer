import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import filters, segmentation, morphology, exposure, restoration, draw
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import pywt
from skimage.filters import gaussian, median, threshold_otsu
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import random_walker
from skimage.filters import laplace
from skimage.feature import canny
from skimage.transform import resize
from sklearn.mixture import GaussianMixture

class Filters():
    def __init__(self):
        super(self).__init__()

    def selectFilter(array, text):
        if text == "Orijinal":
            return array
        elif text == "Thresholded":
            return Filters.apply_threshold(array)
        elif text == "Cleaned":
            return Filters.clean_segmentation(array)
        elif text == "Contour":
            return Filters.create_overlay(array)
        elif text == "Colored":
            return Filters.create_colored_segments(array)
        elif text == "Normalized":
            return Filters.normalize_image(array)
        elif text == "GaussianFilter":
            return Filters.apply_gaussian_filter(array)
        elif text == "HistogramEqualization":
            return Filters.apply_histogram_equalization(array)
        elif text == "AnistropicDiffusion":
            return Filters.apply_anisotropic_diffusion(array)
        elif text == "CLAHE":
            return Filters.apply_clahe(array)
        elif text == "MorphologicalOperations":
            return Filters.apply_morphological_operations(array)
        elif text == "GaussianBlur":
            return Filters.apply_gaussian_blur(array)
        elif text == "FourierFiltering":
            return Filters.apply_fourier_filtering(array)
        elif text == "BilateralFilter":
            return Filters.apply_bilateral_filter(array)
        elif text == "FinalSegmentation":
            return Filters.create_final_segmentation(array)
        elif text == "ROISegmentation":
            return Filters.create_roi_mask(array)
        elif text == "WaveletTransform":
            return Filters.apply_wavelet_transform(array)
        elif text == "BiasFieldCorrection":
            return Filters.apply_bias_correction(array)
        elif text == "Non-LocalMeans":
            return Filters.apply_nlm_denoising(array)
        elif text == "AdaptiveFiltering":
            return Filters.apply_adaptive_filtering(array)
        elif text == "HistogramMatching":
            return Filters.apply_histogram_matching(array)
        elif text == "ResamplingInterpolation":
            return Filters.apply_resampling(array)
        elif text == "LaplacianofGaussian":
            return Filters.apply_log_enhancement(array)
        elif text == "SegmentationSmoothing":
            return Filters.apply_segmentation_smoothing(array)
        elif text == "CannyEdgeDetection":
            return Filters.apply_edge_detection(array)
        elif text == "3DReconstruction":
            return Filters.apply_3d_reconstruction(array)
        elif text == "PhaseCongruency":
            return Filters.apply_phase_congruency(array)
        elif text == "GMMSegmentation":
            return Filters.apply_gmm_segmentation(array)



    def enhance_contrast(image):
        """Enhance image contrast using percentile-based contrast stretching"""
        p2, p98 = np.percentile(image, (2, 98))
        img_contrast = np.clip(image, p2, p98)
        return img_contrast

    # Function to create binary image using Otsu thresholding
    # Input: Grayscale image array
    # Output: Binary image array (True/False values)
    # Method: Automatically determines optimal threshold using Otsu's method
    # - Calculates optimal threshold that minimizes intra-class variance
    # - Returns binary image where True = pixel above threshold
    def apply_threshold(image):
        image = Filters.enhance_contrast(image)
        """Apply Otsu thresholding to create binary image"""
        thresh = filters.threshold_otsu(image)
        binary = image > thresh
        return binary

    # Function to clean up binary segmentation
    # Input: Binary image array
    # Output: Cleaned binary image array
    # Method: Applies morphological operations
    # - Removes small objects below 50 pixels
    # - Performs morphological closing with disk structuring element
    # - Closing fills small holes and connects nearby regions
    def clean_segmentation(binary_image):
        binary_image = Filters.apply_threshold(binary_image)
        """Clean up binary segmentation using morphological operations"""
        cleaned = morphology.remove_small_objects(binary_image, min_size=50)
        cleaned = morphology.closing(cleaned, morphology.disk(3))
        return cleaned

    # Function to find boundaries in binary image
    # Input: Binary image array
    # Output: Binary boundary image array
    # Method: Detects boundaries between regions
    # - Identifies pixels where binary values change
    # - Returns binary image with True at boundary locations
    def find_image_contours(cleaned_image):
        cleaned_image = Filters.clean_segmentation(cleaned_image)
        """Find boundaries/contours in cleaned binary image"""
        contours = segmentation.find_boundaries(cleaned_image)
        return contours

    # Function to overlay contours on original image
    # Input: Original image array and contour array
    # Output: Image array with overlaid contours
    # Method: Combines original image with contours
    # - Creates copy of original image
    # - Sets contour pixels to maximum intensity value
    def create_overlay(image, contours=None):
        contours = image
        image = Filters.enhance_contrast(image)
        contours = Filters.find_image_contours(contours)
        """Create overlay of original image with contours"""
        overlay = image.copy()
        overlay[contours] = np.max(image)
        return overlay

    # Function to create colored segmentation visualization
    # Input: Original image array and binary mask array
    # Output: RGB image array with colored segments
    # Method: Creates color visualization
    # - Converts grayscale to RGB
    # - Colors segmented regions in red
    # - Keeps background in grayscale
    def create_colored_segments(image, cleaned_mask=None):
        cleaned_mask = image
        image = Filters.enhance_contrast(image)
        cleaned_mask = Filters.clean_segmentation(cleaned_mask)
        """Create colored segmentation visualization"""
        colored = np.zeros((*image.shape, 3))
        normalized = image/np.max(image)
        colored[..., 0] = normalized
        colored[..., 1] = normalized 
        colored[..., 2] = normalized
        colored[cleaned_mask, 0] = 1.0
        colored[cleaned_mask, 1] = 0.0
        colored[cleaned_mask, 2] = 0.0
        return colored
    



    # Normalizes the image to 0-1 range
    # Input: Raw image
    # Output: Image normalized to 0-1 range
    def normalize_image(img):
        """Normalize image to 0-1 range"""
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img_norm

    # Applies Gaussian filter to reduce noise
    # Input: Image
    # Output: Noise reduced image
    # sigma=1 parameter determines the filter's effect area
    def apply_gaussian_filter(img):
        img = Filters.normalize_image(img)
        """Apply Gaussian filtering for noise reduction"""
        return ndimage.gaussian_filter(img, sigma=1)

    # Applies median filter to reduce noise
    # Input: Image
    # Output: Noise reduced image
    # size=3 parameter uses a 3x3 window
    def apply_median_filter(img):
        img = Filters.normalize_image(img)
        """Apply Median filtering for noise reduction"""
        return ndimage.median_filter(img, size=3)

    # Applies histogram equalization to improve contrast
    # Input: Image
    # Output: Contrast enhanced image
    def apply_histogram_equalization(img):
        img = Filters.normalize_image(img)
        """Apply histogram equalization"""
        return exposure.equalize_hist(img)

    # Applies anisotropic diffusion filter (Non-local means)
    # Input: Image
    # Output: Edge-preserving noise reduced image
    def apply_anisotropic_diffusion(img):
        img = Filters.normalize_image(img)
        """Apply anisotropic diffusion using non-local means"""
        return restoration.denoise_nl_means(img)

    # Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # Input: Image
    # Output: Local contrast enhanced image
    def apply_clahe(img):
        img = Filters.normalize_image(img)
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        return exposure.equalize_adapthist(img)

    # Applies morphological operations to clean segmentation
    # Input: Image
    # Output: Cleaned binary image
    # Applies Otsu thresholding, closing and small object removal
    def apply_morphological_operations(img):
        img = Filters.apply_clahe(img)
        """Apply morphological operations for cleaning"""
        thresh = filters.threshold_otsu(img)
        binary = img > thresh
        cleaned = morphology.closing(binary, morphology.disk(3))
        cleaned = morphology.remove_small_objects(cleaned, min_size=50)
        return cleaned

    # Applies Gaussian blurring
    # Input: Image
    # Output: Smoothed image
    # sigma=0.5 parameter determines the blurring degree
    def apply_gaussian_blur(img):
        img = Filters.apply_clahe(img)
        """Apply Gaussian blurring"""
        return ndimage.gaussian_filter(img, sigma=0.5)

    # Applies filtering in frequency domain using Fourier transform
    # Input: Image
    # Output: Frequency domain filtered image
    # Masks central 60x60 area to preserve high frequencies
    def apply_fourier_filtering(img):
        img = Filters.normalize_image(img)
        """Apply Fourier transform filtering"""
        f = fft2(img)
        fshift = fftshift(f)
        rows, cols = img.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols))
        mask[crow-30:crow+30, ccol-30:ccol+30] = 0
        fshift = fshift * mask
        f_ishift = ifftshift(fshift)
        return np.abs(ifft2(f_ishift))

    # Applies bilateral filtering
    # Input: Image
    # Output: Edge-preserving smoothed image
    def apply_bilateral_filter(img):
        img = Filters.normalize_image(img)
        """Apply bilateral filtering"""
        return restoration.denoise_bilateral(img)

    # Performs final segmentation
    # Input: Image
    # Output: Cleaned binary segmentation image
    # Applies Otsu thresholding, small object removal and closing
    def create_final_segmentation(img):
        img = Filters.apply_bilateral_filter(img)
        """Create final segmentation"""
        thresh = filters.threshold_otsu(img)
        binary = img > thresh
        cleaned = morphology.remove_small_objects(binary, min_size=50)
        cleaned = morphology.closing(cleaned, morphology.disk(3))
        return cleaned

    # Creates and applies Region of Interest (ROI) mask
    # Input: Original image and cleaned segmentation
    # Output: Segmentation with applied ROI mask
    # Creates circular ROI in image center
    def create_roi_mask(img, cleaned=None):
        cleaned = Filters.create_final_segmentation(img)
        """Create and apply ROI mask"""
        center_y, center_x = img.shape[0]//2, img.shape[1]//2
        radius = min(center_y, center_x) // 2
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        roi_mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)
        roi_result = cleaned.copy()
        roi_result[~roi_mask] = 0
        return roi_result
    




    # Applies Haar wavelet transform
    # Input: Image
    # Output: Wavelet transformed image
    # Decomposes image into different frequency components
    def apply_wavelet_transform(img):
        img = Filters.normalize_image(img)
        coeffs = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs
        img_wavelet = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        return img_wavelet

    # Applies bias field correction
    # Input: Image
    # Output: Bias corrected image
    # Uses CLAHE with clip_limit=0.02 to correct intensity inhomogeneities
    def apply_bias_correction(img):
        img = Filters.normalize_image(img)
        img_bias = exposure.equalize_adapthist(img, clip_limit=0.02)
        return img_bias

    # Applies Non-Local Means denoising
    # Input: Image
    # Output: Denoised image
    # Estimates noise and applies NLM with h=1.15*sigma
    def apply_nlm_denoising(img):
        img = Filters.normalize_image(img)
        sigma_est = np.mean(estimate_sigma(img))
        img_nlm = denoise_nl_means(img, h=1.15 * sigma_est)
        return img_nlm

    # Applies adaptive median filtering
    # Input: Image
    # Output: Filtered image
    # Uses disk structuring element with radius=3
    def apply_adaptive_filtering(img):
        img = Filters.normalize_image(img)
        img_adaptive = median(img, morphology.disk(3))
        return img_adaptive

    # Applies histogram matching !!!!!
    # Input: Image and reference image
    # Output: Matched image
    # Matches histogram of input to reference image
    def apply_histogram_matching(img, reference=None):
        reference = img
        img_matched = exposure.match_histograms(img, reference)
        return img_matched

    # Applies resampling and interpolation
    # Input: Image
    # Output: Resampled image
    # Upsamples by 2x then downsamples back with antialiasing
    def apply_resampling(img):
        img = Filters.normalize_image(img)
        img_up = resize(img, (img.shape[0]*2, img.shape[1]*2), anti_aliasing=True)
        img_resampled = resize(img_up, img.shape, anti_aliasing=True)
        return img_resampled

    # Applies Laplacian of Gaussian enhancement
    # Input: Image
    # Output: Enhanced image
    # Uses sigma=2 for Gaussian and normalizes output
    def apply_log_enhancement(img):
        img = Filters.normalize_image(img)
        img_log = ndimage.gaussian_laplace(img, sigma=2)
        img_log = (img_log - np.min(img_log)) / (np.max(img_log) - np.min(img_log))
        return img_log

    # Applies segmentation smoothing
    # Input: Image
    # Output: Smoothed binary image
    # Uses Otsu thresholding and morphological closing
    def apply_segmentation_smoothing(img):
        img = Filters.normalize_image(img)
        thresh = threshold_otsu(img)
        binary = img > thresh
        img_smooth = morphology.binary_closing(binary)
        img_smooth_seg = gaussian(img_smooth, sigma=1)
        return img_smooth_seg

    # Applies Canny edge detection
    # Input: Image
    # Output: Edge image
    # Uses sigma=2 for Gaussian smoothing
    def apply_edge_detection(img):
        img = Filters.normalize_image(img)
        img_edges = canny(img, sigma=2)
        return img_edges

    # Applies 3D reconstruction simulation
    # Input: Image
    # Output: Reconstructed image
    # Uses zoom factor=1.0 for demonstration
    def apply_3d_reconstruction(img):
        img = Filters.normalize_image(img)
        img_3d = ndimage.zoom(img, 1.0)
        return img_3d

    # Applies phase congruency approximation
    # Input: Image
    # Output: Phase image
    # Uses Sobel filter to detect edges
    def apply_phase_congruency(img):
        img = Filters.normalize_image(img)
        img_phase = filters.sobel(img)
        return img_phase

    # Applies GMM-based segmentation
    # Input: Image
    # Output: Segmented image
    # Uses 3 components for clustering
    def apply_gmm_segmentation(img):
        img = Filters.normalize_image(img)
        X = img.reshape(-1, 1)
        gmm = GaussianMixture(n_components=3, random_state=42)
        img_gmm = gmm.fit_predict(X).reshape(img.shape)
        return img_gmm