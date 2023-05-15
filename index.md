#### Image Processing problems

- Image Restoration

  - de-noising

  - de-blurring(sharpening)

- Image Compression

  - Jpeg, HEIF, MPEG

- Computing Field Properties



Image Filtering

- Modify the pixels in an image based on some function of a local neighborhood of each pixel
- One simple version of filtering: linear filtering(cross-correlation, convolution)
  - Replace each pixel by a linear combination(a weighted sum) of its neighbors
- The prescription for the linear combination is called the "Kernel" (or "mask", "filter")



#### Cross-correlation

Let F be the image, H be the kernel (of size 2k+1 * 2k + 1) and G be the output Image

#### Convolution

- G = H * F

- Convolution is commutative and associative

Mean filtering / Moving average



Gaussian Kernel

Gaussian filters

- Removes "high-frequency" components from the image(low-pass filter)
- Convolution with self is another Gaussian



Sharpening revisited

- What does blurring take away?
  - origin - smoothed(5*5)
  - This "detail extraction" operation is also called a high-pass filter
  - original + a * detail = sharpened
- F + a * (F - F * H) = (1 + a) * F - a(F * H) = F * ([1+a] * e - a * H)
  - e: unit impulse (identity kernel with single 1 in center, zeros elsewhere)
  - F: image
  - F * H: blurred image
  - scaled impulse - Gaussian = sharpen filter



Image sub-sampling

Throw away every other row and column to create a 1/2 size image.



Aliasing

- Occurs when your sampling rate is not high enough to capture the amount of detail in your image
- Can give you the wrong signal / image - an alias
- to do sampling right, need to understand the structure of your signal/image
- Enter Monsieur Fourier...
- To avoid aliasing
  - sampling rate >= 2* max frequency in the image
    - said another way: >= two per cycle7
  - This minimum sampling rate is called the Nyquist rate



Upsampling

- This image is to small for this screen
- How can we make it 10 times as big?
- Simplest approach: repeat each row and column 10 times
- ("Nearest neighbor interpolation")

Image interpolation

- Recall the a digital images is formed as follows:

  - F[x, y] = quantize{f(xd,yd)}

  - It is a discrete point-sampling of a continuous function
  - If we could somehow reconstruct the original function, any new image could be generated, at any resolution and scale

- What if we don't know f ?

  - Guess an approximation: f
  - Can be done in a principled way: filtering
  - Convert F to a continuous function
    - f~F~(x)=F(x/d) when x/d is an integer, 0 otherwise
  - Reconstruct by convolution with a reconstruction filter, h
    - f = h * f~F~

Super - resolution with multiple images

- Can do better upsampling of you have multiple images of the scene taken with small(subpixel) shifts
- Some cellphone cameras (like the google pixel line) capture a burst of photos





### 20221012

Characterizing edges

- An edge is a place of rapid change in the image intensity function.
- First derivative: edges correspond to extrema of derivative

Image derivative

- How can we differentiate a digital image F[x,y]?
  - Reconstruct a continuous image, f,f then computed the derivative
  - take discrete derivative(finite difference)
- And both what happens when you have an edge is that direction the next when you get to another pixel, they' ll be a big change in that direction. And that's what tells us that we' ve got to match.
  So what have we got here?

Solution: smooth first

- f, signal sigma=50
- $d/dx * h$
- $f * d/dx * h$

Associative property of convolution

- Differentiation is convolution, and convolution is associative: $d/dx(f*h) = f * d/dx*h$

The sobel operator

- common approximation of derivative of Gaussian
- The standard definition of the sobel operator omits the 1/8 term
  - does not make a difference for edge detection
  - the 1/8 term is needed to get

Canny edge detector

- Filter image with derivative of Gaussian
- Find magnitude and orientaion of gradient
- Non-maximum suppression
- Linking and thresholding(hysteresis):
  - Define two thresholds: low and high
  - Use the high threshold to start edge curves and the low threshold to continue them





### 20221019

Application: Visual SLAM

- aka Simultaneous Localization and Mapping

Image matching

Feature matching for object

Invariant local features

- find features that are invariant to transformations
  - geometric invariance: translation, rotation, scale
  - photometric invariance: brightness, exposure.

Advantages of local features

- Locality
  - features are local, so robust to occlusion and clutter
- QUantity
  - hundreds or thousands in a single image
- Distinctiveness:
  - can differentiate a large database of ovjects
- Effieciency
  - real-time

More motivation

- feature points are used for
  - image alignments(e.g. mosaics)
  - 3D reconstruction
  - Motion tracking(e.g AR)
  - Object recognition
  - image retrieval
  - robot/cat navigation
  - other...

Approach

1. Feature detection: find it
2. Feature descriptor
3. Feature matching
4. Feature tracking

Local features: main components

1. Detection: identify the interest points
2. Description: Extract vector feature descriptor surrounding each interest point
3. Matching: determine correspondence between descriptors in two views

What makes a good feature?

- Want uniqueness - look for image regions that are unusual
  - lead to unambiguous matches in other images
- how to define "unusual"
  - Suppose we only consider a small window of pixels
    - What defines whether a feature is a good or bad candidate?
  - "Flat" region - no change in all directions
  - "edge" region - no change
  - "corner"

Harris corner detection: the math

- consider shifting the window W by(u,v)
  - how do the pixels in W change?
  - compare each pixel before and after by summing up the squared differences
  - the defines an SSD "error" E(u,v)
  - we are happy if this error is high
  - slow to compute exactly for each pixel and each offset(u, v)

Small motion assumption

- Taylor series expansion of I:
  - $I(x+u, y+v) = I(x, y) + eI/ex * u + eI/ey * v + higher\_order\_terms$



### Recognition

Why not use SIFT matching for everything?

- Works well for object instances (or distinctive images such as logos)
  - pepsi, cocacola
- Not great for generic object categories

Applications:

- Photography
- shutter-free photography(take photos when you smile)
- photo organization

**Why is recognition hard?**

- Variability:
  - Camera position
  - Illumination,
  - Shape
  - etc...

What Matters in Recognition?

- Learning Techniques
  - E.g. choice of classifier or inference method
- Representation
  - Low level: SIFT, HoG, GIST, edges
  - Mid Level: Bag of words, sliding window, deformable model
  - High level: contextual dependence
  - Deep learned features
- Data
  - More is always better(as long as it is good data)
  - Annotation is the hard part
- 