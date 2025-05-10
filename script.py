import cv2
import numpy as np
from functools import lru_cache
import os
import time
from concurrent.futures import ThreadPoolExecutor

class FlagPatternMapper:
    """Maps a pattern image onto a flag using mesh warping techniques.
    
    This class provides functionality to transform pattern images onto flag
    surfaces using mesh warping techniques from computer vision. It supports
    both file paths and numpy arrays as inputs.
    
    Attributes:
        pattern_img (np.ndarray): The loaded pattern image
        flag_img (np.ndarray): The loaded flag image
        pattern_pts (np.ndarray): Grid points on the pattern
        flag_pts (np.ndarray): Corresponding points on the flag 
        triangles (list): Triangulation data for mesh warping
        processed_pattern (np.ndarray): Cached preprocessed pattern
        flag_mask (np.ndarray): Cached flag mask
    """
    
    def __init__(self, pattern_input, flag_input):
        """Initialize the FlagPatternMapper with pattern and flag inputs.
        
        Args:
            pattern_input (str or np.ndarray): Path to pattern image or image array
            flag_input (str or np.ndarray): Path to flag image or image array
            
        Raises:
            ValueError: If images cannot be loaded or processed
        """
        self.pattern_img, self.flag_img = self._load_images(pattern_input, flag_input)
        if self.pattern_img is None or self.flag_img is None:
            raise ValueError("Failed to load images")
        
        # Pre-calculations with optimization
        self.pattern_pts = self._generate_grid_points()
        self.flag_pts = self._get_flag_points()
        self.triangles = self._generate_triangles()
        
        # Cached computations
        self.processed_pattern = None
        self.flag_mask = None

    @staticmethod
    def _load_images(pattern_input, flag_input):
        """Load images from paths or use numpy arrays directly.
        
        This method handles different input types (file paths or numpy arrays)
        and returns properly formatted image arrays. It ensures that pattern
        and flag images have compatible dimensions.
        
        Args:
            pattern_input (str or np.ndarray): Path to pattern image or image array
            flag_input (str or np.ndarray): Path to flag image or image array
            
        Returns:
            tuple: (pattern_img, flag_img) as numpy arrays
            
        Raises:
            ValueError: If images cannot be loaded or are invalid
        """
        try:
            # Handle pattern input
            if isinstance(pattern_input, str):
                pattern = cv2.imread(pattern_input)
                if pattern is None:
                    raise ValueError(f"Failed to load pattern image: {pattern_input}")
            elif isinstance(pattern_input, np.ndarray):
                pattern = pattern_input.copy()
            else:
                raise ValueError("Pattern input must be a file path or numpy array")
                
            # Handle flag input
            if isinstance(flag_input, str):
                flag = cv2.imread(flag_input)
                if flag is None:
                    raise ValueError(f"Failed to load flag image: {flag_input}")
            elif isinstance(flag_input, np.ndarray):
                flag = flag_input.copy()
            else:
                raise ValueError("Flag input must be a file path or numpy array")
                
            # Resize pattern to match flag dimensions
            h_flag, w_flag = flag.shape[:2]
            pattern = cv2.resize(pattern, (w_flag, h_flag))
                
            return pattern, flag
        except Exception as e:
            print(f"Error loading images: {str(e)}")
            return None, None

    @lru_cache(maxsize=8)
    def _generate_grid_points(self, rows=5, cols=5):
        """Generate a grid of control points for the pattern image.
        
        Creates an evenly spaced grid of points on the pattern image
        that will be used as source points for the mesh warping process.
        Results are cached for performance.
        
        Args:
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
            
        Returns:
            np.ndarray: Array of grid points with shape (rows*cols, 2)
        """
        h, w = self.pattern_img.shape[:2]
        xs = np.linspace(0, w-1, cols)
        ys = np.linspace(0, h-1, rows)
        return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)

    @lru_cache(maxsize=1)
    def _get_flag_points(self):
        """Get the predefined control points on the flag surface.
        
        Returns fixed destination points on the flag that correspond
        to the grid points on the pattern. These points are designed
        to follow the natural contours of the flag surface.
        Results are cached for performance.
        
        Returns:
            np.ndarray: Array of flag control points with shape (25, 2)
        """
        return np.array([
            [290, 74], [514, 160], [702, 228], [874, 244], [1046, 254],
            [286, 250], [496, 306], [690, 360], [852, 390], [1012, 416],
            [286, 408], [492, 448], [670, 480], [846, 508], [986, 534],
            [288, 536], [468, 564], [662, 584], [828, 616], [978, 646],
            [292, 676], [478, 682], [654, 694], [822, 784], [986, 814]
        ], dtype=np.float32)

    @lru_cache(maxsize=1)
    def _generate_triangles(self, rows=5, cols=5):
        """Generate triangulation data for the mesh warping.
        
        Creates triangle indices from the grid points. Each grid cell
        is divided into two triangles. These triangles define the mesh
        structure used for warping the pattern onto the flag.
        Results are cached for performance.
        
        Args:
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
            
        Returns:
            list: List of triangle vertex indices, each as [i1, i2, i3]
        """
        triangles = []
        for i in range(rows-1):
            for j in range(cols-1):
                idx = i * cols + j
                triangles.append([idx, idx+1, idx+cols])
                triangles.append([idx+1, idx+cols+1, idx+cols])
        return triangles

    def create_flag_mask(self):
        """Create a mask of the flag's surface for blending.
        
        Generates a mask defining the flag area where the pattern will be applied.
        The mask is created by filling all triangles in the flag mesh and then
        applying a Gaussian blur for smooth edges. Results are cached.
        
        Returns:
            np.ndarray: Flag mask as a float32 array with values in [0,1]
        """
        if self.flag_mask is not None:
            return self.flag_mask
            
        mask = np.zeros(self.flag_img.shape[:2], dtype=np.uint8)
        
        for tri_indices in self.triangles:
            triangle = self.flag_pts[tri_indices].astype(np.int32)
            cv2.fillConvexPoly(mask, triangle, 255)
        
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        self.flag_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        return self.flag_mask

    def preprocess_pattern(self):
        """Preprocess the pattern image for improved warping results.
        
        Enhances the pattern image with:
        - Contrast adjustment
        - Gaussian blur for smoothing
        - Edge handling with a mask to prevent artifacts
        - Multiple blur levels for different regions
        
        Results are cached for performance.
        
        Returns:
            np.ndarray: Enhanced pattern image ready for warping
        """
        if self.processed_pattern is not None:
            return self.processed_pattern
            
        try:
            pattern = cv2.convertScaleAbs(self.pattern_img, alpha=1.2, beta=20)
            pattern = cv2.GaussianBlur(pattern, (3, 3), 0)
            
            h, w = pattern.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            corner_margin = 25
            cv2.rectangle(mask, 
                        (corner_margin, corner_margin), 
                        (w-corner_margin, h-corner_margin), 
                        255, -1)
            
            mask = cv2.GaussianBlur(mask, (41, 41), 0)
            mask = mask[:, :, np.newaxis] / 255.0
            
            pattern_blur1 = cv2.GaussianBlur(pattern, (7, 7), 0)
            pattern_blur2 = cv2.GaussianBlur(pattern, (15, 15), 0)
            
            self.processed_pattern = pattern * mask + pattern_blur1 * (1-mask)*0.7 + pattern_blur2*(1-mask)*0.3
            return self.processed_pattern
            
        except Exception as e:
            print(f"Error preprocessing pattern: {str(e)}")
            self.processed_pattern = self.pattern_img.copy()
            return self.processed_pattern

    def _warp_triangle(self, src_img, src_tri, dst_tri, output_img):
        """Warp a single triangle from the pattern to the flag.
        
        Performs the core warping operation for a single triangle:
        1. Calculate bounding rectangles for source and destination triangles
        2. Extract the source triangle region with padding
        3. Compute the affine transformation matrix
        4. Warp the source to the destination shape
        5. Create a mask for smooth blending
        6. Blend the warped triangle into the output image
        
        Args:
            src_img (np.ndarray): Source pattern image
            src_tri (np.ndarray): Source triangle coordinates [(x1,y1), (x2,y2), (x3,y3)]
            dst_tri (np.ndarray): Destination triangle coordinates
            output_img (np.ndarray): Output image to blend the warped triangle into
            
        Note:
            This function modifies output_img in-place
        """
        try:
            r_src = cv2.boundingRect(np.float32([src_tri]))
            r_dst = cv2.boundingRect(np.float32([dst_tri]))

            src_offset = src_tri - r_src[:2]
            dst_offset = dst_tri - r_dst[:2]

            pad = 1
            x1 = max(0, r_src[0]-pad)
            y1 = max(0, r_src[1]-pad)
            x2 = min(src_img.shape[1], r_src[0]+r_src[2]+pad)
            y2 = min(src_img.shape[0], r_src[1]+r_src[3]+pad)
            src_crop = src_img[y1:y2, x1:x2]
            
            if r_src[1]-pad < 0:
                src_offset[:,1] += abs(r_src[1]-pad)
            if r_src[0]-pad < 0:
                src_offset[:,0] += abs(r_src[0]-pad)

            M = cv2.getAffineTransform(src_offset.astype(np.float32), dst_offset.astype(np.float32))
            
            x1_dst = max(0, r_dst[0]-pad)
            y1_dst = max(0, r_dst[1]-pad)
            x2_dst = min(output_img.shape[1], r_dst[0]+r_dst[2]+pad)
            y2_dst = min(output_img.shape[0], r_dst[1]+r_dst[3]+pad)
            
            warp_width = x2_dst - x1_dst
            warp_height = y2_dst - y1_dst
            
            if warp_width <= 0 or warp_height <= 0:
                return
                
            warped_patch = cv2.warpAffine(
                src_crop, M, (warp_width, warp_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            
            mask = np.zeros((warp_height, warp_width, 3), dtype=np.float32)
            roi_offset = dst_offset.copy()
            
            if r_dst[0]-pad < 0:
                roi_offset[:,0] += abs(r_dst[0]-pad)
            if r_dst[1]-pad < 0:
                roi_offset[:,1] += abs(r_dst[1]-pad)
            
            cv2.fillConvexPoly(mask, np.int32(roi_offset), (1, 1, 1))
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            roi = output_img[y1_dst:y2_dst, x1_dst:x2_dst]
            if roi.shape == mask.shape == warped_patch.shape:
                output_img[y1_dst:y2_dst, x1_dst:x2_dst] = \
                    roi * (1 - mask) + warped_patch * mask
        
        except Exception as e:
            pass

    def create_shading_mask(self, flag_mask):
        """Create a shading mask for realistic lighting effects.
        
        Analyzes the flag image to create realistic lighting/shading effects:
        1. Convert to grayscale and enhance contrast
        2. Calculate surface gradients using Sobel operators
        3. Compute gradient magnitude for shading intensity
        4. Normalize and scale for visual appeal
        5. Combine with flag mask for proper region application
        
        Args:
            flag_mask (np.ndarray): Flag mask defining the application region
            
        Returns:
            np.ndarray: Shading mask as a float32 array with values in [0,1]
        """
        try:
            gray = cv2.cvtColor(self.flag_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            displacement = cv2.GaussianBlur(enhanced, (15,15), 0).astype(np.float32) / 255.0
            dx = cv2.Sobel(displacement, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(displacement, cv2.CV_32F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
            mean_magnitude = np.mean(gradient_magnitude)
            
            shading = 0.5 + 0.7 * (gradient_magnitude - mean_magnitude)
            shading = np.clip(shading, 0.4, 1.0)
            
            shading = cv2.cvtColor(shading.astype(np.float32), cv2.COLOR_GRAY2BGR)
            return shading * flag_mask + (1 - flag_mask)
            
        except Exception as e:
            print(f"Error creating shading mask: {str(e)}")
            return np.ones_like(flag_mask)

    def warp_mesh(self):
        """Perform the complete mesh warping operation.
        
        The main warping process that:
        1. Gets the preprocessed pattern
        2. Creates an empty output image
        3. For each triangle in the mesh:
           - Maps the triangle from pattern to flag
           - Blends the warped triangle into the output
        4. Optimizes performance with parallel processing for many triangles
        
        Returns:
            np.ndarray: Warped pattern image
        """
        processed_pattern = self.preprocess_pattern()
        warped = np.zeros_like(self.flag_img)
        
        max_workers = min(os.cpu_count() or 4, 8)
        
        if len(self.triangles) < 10:
            for tri_indices in self.triangles:
                src_tri = self.pattern_pts[tri_indices]
                dst_tri = self.flag_pts[tri_indices]
                self._warp_triangle(processed_pattern, src_tri, dst_tri, warped)
        else:
            batch_size = max(1, len(self.triangles) // max_workers)
            triangle_batches = [self.triangles[i:i + batch_size] for i in range(0, len(self.triangles), batch_size)]
            
            def process_batch(batch):
                local_warped = np.zeros_like(self.flag_img)
                for tri_indices in batch:
                    src_tri = self.pattern_pts[tri_indices]
                    dst_tri = self.flag_pts[tri_indices]
                    self._warp_triangle(processed_pattern, src_tri, dst_tri, local_warped)
                return local_warped
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_batch, triangle_batches))
            
            for batch_result in results:
                np.maximum(warped, batch_result, out=warped)
        
        return warped

    def blend_with_shading(self, warped_img, flag_mask):
        """Blend the warped image with the flag using shading effects.
        
        Performs the final blending steps:
        1. Convert images to float for precision
        2. Apply the flag mask to blend warped pattern and flag
        3. Apply shading mask for realistic lighting
        4. Blend again with the flag for final composition
        5. Clip values to valid range and convert back to uint8
        
        Args:
            warped_img (np.ndarray): Warped pattern image
            flag_mask (np.ndarray): Flag mask defining the application region
            
        Returns:
            np.ndarray: Final blended image with shading effects
        """
        try:
            flag_img_float = self.flag_img.astype(np.float32)
            warped_img_float = warped_img.astype(np.float32)
            shading_mask = self.create_shading_mask(flag_mask)
            
            blended = flag_img_float * (1 - flag_mask) + warped_img_float * flag_mask
            blended = blended * shading_mask
            final = blended * flag_mask + flag_img_float * (1 - flag_mask)
            
            return np.clip(final, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Error in blending: {str(e)}")
            return self.flag_img.copy()

    def process(self, output_path=None):
        """Execute the complete image processing pipeline.
        
        Orchestrates the full pattern-to-flag mapping process:
        1. Warp the pattern mesh to match the flag surface
        2. Create the flag mask for blending
        3. Apply shading and blending effects
        4. Optionally save the result to disk
        5. Return the processed image
        
        Args:
            output_path (str, optional): Path to save output image.
                If None, image is not saved.
            
        Returns:
            np.ndarray: Final processed image or None if processing fails
            
        Raises:
            Exception: If any step in the processing pipeline fails
        """
        try:
            start_time = time.time()
            warped = self.warp_mesh()
            flag_mask = self.create_flag_mask()
            final_result = self.blend_with_shading(warped, flag_mask)
            
            if output_path:
                cv2.imwrite(output_path, final_result)
                print(f"Processing completed in {time.time() - start_time:.2f} seconds")
            
            return final_result
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return None


def process_image(pattern_img, flag_img):
    """Process pattern and flag images to create a mapped result.
    
    This is the main interface function called by the Streamlit application.
    It creates a FlagPatternMapper instance and processes the images.
    
    Args:
        pattern_img (np.ndarray): Pattern image as numpy array
        flag_img (np.ndarray): Flag image as numpy array
        
    Returns:
        np.ndarray: Processed image or None if processing fails
        
    Example:
        >>> result = process_image(pattern_array, flag_array)
        >>> cv2.imwrite("result.jpg", result)
    """
    try:
        mapper = FlagPatternMapper(pattern_img, flag_img)
        return mapper.process()
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None