# Image compression with k-means

<p> This projects load an images, encodes it using k-means then decodes it using the transmitted data </p>

* There are two important parameters
  * c: What will be the size of clustered area (cxc)
  * k: How many 1xc vectors will be used to represent all vectors

**Compression rate** is computed as follows (each pixel is assumed to be represented in 8 bits)  

**Original image size:** n x n x 8 x 3  
**Compressed image size:** ((n/c)^2 x log_2(k) + k x c x c x 8) x 3  

**Compression rate = (Original image size)/(Compressed image size)**

## Examples

**k=16, c=2**  
**Compression rate = 7**

![compressed_16_2] (images/result_k_16_c_2.png)

**k=8, c=2**  
**Compression rate = 10**



**k=8, c=4**  
**Compression rate = 39**



**k=4, c=2**  
**Compression rate = 15**
