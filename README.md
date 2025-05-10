
# Flag Pattern Mapper
## Prerequisites
- Docker or Python 3.10+
- For native Python installation:
  - OpenCV
  - NumPy
  - Streamlit
  - Pillow

## Installation

### Using Docker 

1. Clone the repository:
   ```bash
   git clone https://github.com/sanmita13742/flag-pattern-mapper.git
   cd flag-pattern-mapper
   ```

2. Build the Docker image:
   ```bash
   docker build -t flag-pattern-mapper .
   ```

3. Run the container:
   ```bash
   docker run -p 8501:8501 flag-pattern-mapper
   ```

4. Access the app at: [http://localhost:8501](http://localhost:8501)

### Native Python Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sanmita13742/flag-pattern-mapper
   cd flag-pattern-mapper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload a pattern image** (or use the sample grid pattern)
2. Click **"Process Image"** to apply the pattern to the flag
3. View and **download your result**

## Project Structure

```
flag-pattern-mapper/
├── Dockerfile             # Docker configuration
├── requirements.txt      # Python dependencies
├── app.py                # Streamlit application
├── script.py             # Core image processing logic
├── sample_images/        # Sample images
│   └── flag.png          # Default flag image
└── README.md             # This file
```

# Flag Pattern Mapper: How It Works

## Overview

The Flag Pattern Mapper is a Streamlit web application that maps pattern images onto flag images using mesh warping techniques from computer vision. This document explains the technical approach and provides insights into the implementation.

## Application Flow

1. **User Interface (Streamlit)**
    
    - Users upload a pattern image or use the default sample
    - The application displays both the pattern and flag images
    - Clicking "Process Image" triggers the transformation process
    - The result is displayed with a download option
2. **Processing Pipeline**
    
    - Pattern preprocessing (contrast enhancement, edge handling)
    - Mesh generation and warping
    - Blending with lighting/shading effects for realism
    - Final compositing of the warped pattern with the flag

## Technical Approach: Mesh Warping

### 1. Grid and Control Points

- The pattern image is divided into a grid (5×5 points)
- Each grid point is mapped to a corresponding predefined point on the flag
- These predefined points follow the natural contours of the flag

### 2. Triangulation

- The grid points are connected to form triangles
- Each triangle in the pattern corresponds to a triangle on the flag
- This creates the mesh structure needed for warping

### 3. Triangle Warping

For each triangle:

- An affine transformation matrix is calculated
- The pattern within the triangle is warped to match the flag's triangle
- Each warped triangle is blended with a smooth mask

### 4. Realistic Shading

- Gradient analysis is performed on the flag image
- A shading mask is created based on natural light directions
- This adds realistic lighting effects to the warped pattern

### 5. Final Blending

- The warped pattern is blended with the flag using the flag mask
- Shading effects are applied for realism
- Final compositing ensures smooth transitions

## Performance Optimizations

1. **Caching with `@lru_cache`**
    
    - Grid and triangle generation are cached
    - Preprocessed patterns and masks are cached
2. **Parallel Processing**
    
    - Triangle warping is done in parallel using ThreadPoolExecutor
    - Each worker processes a batch of triangles
    - Results are combined using numpy's fast operations
3. **Efficient Image Operations**
    
    - OpenCV's optimized functions are used
    - In-place operations minimize memory usage
    - Bounding rectangles are used to process only necessary pixels
