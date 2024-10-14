from numpy.fft import fft, ifftshift, ifft
import numpy as np

def phasecongmono(img, nscale: int = 4, minwavelength: float = 3, 
                  mult: float = 2.1, sigmaonf: float = 0.55, k: float = 3.0,
                  noisemethod: float = -1, cutoff: float = 0.5, g: float = 10.0,
                  deviationgain: float = 1.5 ):
    
    epsilon         = .0001            # Used to prevent division by zero.

    (rows,cols) = img.size

    IMG = fft(img)                     # Use fft rather than perfft2

    sumAn  = np.zeros((rows,cols))         # Accumulators
    sumf   = np.zeros((rows,cols)) 
    sumh1  = np.zeros((rows,cols)) 
    sumh2  = np.zeros((rows,cols)) 
    maxAn =  np.zeros((rows,cols))          # Need maxAn in main scope of function

    IMGF = np.zeros((rows, cols), dtype=np.complex_)  # Buffers
    h = np.zeros((rows, cols), dtype=np.complex_)
    f = np.zeros((rows,cols)) 
    h1 = np.zeros((rows,cols)) 
    h2 = np.zeros((rows,cols)) 
    An = np.zeros((rows,cols)) 

    or_ = np.zeros((rows,cols))    # Final output arrays
    ft = np.zeros((rows,cols)) 
    energy = np.zeros((rows,cols)) 
    PC = np.zeros((rows,cols)) 

    tau = 0.0
    T = 0.0

    # Generate filter grids in the frequency domain
    H, freq = packedmonogenicfilters(rows,cols)

    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        for n in np.ndindex(freq.shape):
            IMGF[n] = IMG[n] * loggabor(freq[n], fo, sigmaonf) * lowpassfilter(freq[n], 0.45, 15)

        f = np.real(ifft(IMGF))
        h = IMGF * H
        h = ifft(h)

        h1 = np.real(h)
        h2 = np.imag(h)
        An = np.sqrt(f**2 + h1**2 + h2**2)
        sumAn += An
        sumf += f
        sumh1 += h1
        sumh2 += h2

        if s == 0:
            if abs(noisemethod + 1) < epsilon:
                tau = np.median(sumAn) / np.sqrt(np.log(4))
            elif abs(noisemethod + 2) < epsilon:
                tau = rayleighmode(sumAn)
            maxAn = An
        else:
            maxAn = np.maximum(maxAn, An)
    
    width = (sumAn / (maxAn + epsilon) - 1) / (nscale - 1)
    weight = 1.0 / (1.0 + np.exp((cutoff - width) * g))

    if noisemethod >= 0:
        T = noisemethod
    else:
        totalTau = tau * (1 - (1 / mult)**nscale) / (1 - (1 / mult))
        EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
        EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
        T = EstNoiseEnergyMean + k * EstNoiseEnergySigma

    or_ = np.arctan(-sumh2 / sumh1)
    ft = np.arctan2(sumf, np.sqrt(sumh1**2 + sumh2**2))
    energy = np.sqrt(sumf**2 + sumh1**2 + sumh2**2)
    PC = weight * np.maximum(1 - deviationgain * np.arccos(energy / (sumAn + epsilon)), 0) * \
         np.maximum(energy - T, 0) / (energy + epsilon)
    

    return PC, or_, ft, T

def packedmonogenicfilters(rows: int, cols: int):
    # from: https://github.com/peterkovesi/ImagePhaseCongruency.jl/blob/c70af1ac4f2aecf1948f57df5891020861d44ff1/src/frequencyfilt.jl#L198-L231
    
    (f, fx, fy) = filtergrids(rows, cols)
    f[0,0] = 1

    # Pack the two monogenic filters
    H = (1j * fx - fy) / f  # Using 1j to represent the imaginary unit in Python

    H[0, 0] = 0  # Restore 0 DC value
    f[0, 0] = 0  # Set the DC value of f back to 0

    return H, f

def filtergrids(rows: int, cols: int):
    # from https://github.com/peterkovesi/ImagePhaseCongruency.jl/blob/84e45cb20cc2e8e7feea151b29c956c23c4d8a20/src/frequencyfilt.jl#L45-L65

    if cols % 2 == 0: # if even 
        fxrange = np.arange(-cols/2, (cols/2) - 1) / cols
    else: # if odd
        fxrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / cols

    if rows % 2 == 0:
        fyrange = np.arange(-rows/2, (rows/2) - 1) / rows
    else: # if odd
        fyrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / rows
    
    fx = np.tile(fxrange, (len(fyrange), 1))
    fy = np.tile(fyrange, (len(fyrange), 1))

    fx = ifftshift(fx)
    fy = ifftshift(fy)

    f = np.sqrt(fx**2 + fy**2)

    return f, fx, fy
    