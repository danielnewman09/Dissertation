import numpy as np
from matplotlib import pyplot as plt


#----- Utility Functions ------------------------------------------------------
def digseq(seq,step):
    '''Original MATLAB preamble
        digseq - Whit Rappole
        DIGITIZESEQ Map a sequence onto digital timing loop
        dseq = digseq(seq,step)

        Uses a linear extrapolation to split each continuous
        impulse into two digital impulses


    Converted to Python on 2/18/13 by Joshua Vaughan (joshua.vaughan@louisiana.edu)'''

    dseq = np.zeros((int(round(seq[-1,0]/step))+2,1))

    for nn in range(len(seq)):
        index = int(np.floor(seq[nn,0]/step))
        woof = (seq[nn,0]-index*step)/step
        dseq[index+1] = dseq[index+1] + woof*seq[nn,1]
        dseq[index] = dseq[index]+seq[nn,1] - woof*seq[nn,1]

    dseq = np.trim_zeros(dseq,'b')
    #while dseq[len(dseq)-1] == 0:
    #    dseq = dseq[0:(len(dseq)-1)]

    return dseq


def conv(Input,Shaper,deltaT):
    """Original MATLAB preamble
        Convolve(Input,Shaper,deltaT) -- Bill Singhose

        function [T,ShapedInput] = convolve(Input,Shaper,deltaT)

        Covolves Input and Shaper and returns the result ShapedInput.
        A time vector, T, which has the same number of rows as
        ShapedInput is also returned. T starts at zero and is incremented
        deltaT each step.
        Input can be an nxm matrix,where m is the number of inputs.
        Shaper must be a row or column vector.


    Converted to Python on 2/19/13 by Joshua Vaughan (joshua.vaughan@louisiana.edu)
    """


    # Just one column for now - JEV - 2/19/13
    if np.size(Input.shape) == 1:
        columns = 0
        rows = Input.shape[0]
    else:
        (rows,columns)=Input.shape

    shlen=len(Shaper)

    # Pad the Input vector with j extra final values,
    #   where j is the length of the shaper.
    for jj in range(columns):
        if columns == 0:
            Input = np.append(Input,[Input[-1]*ones((shlen))])
        else:
            # Input[rows+j,jj]=Input[rows,jj]
            Input = np.append(Input,Input[rows-1,-1]*np.ones((shlen,columns)),0)

    # Reshape into vectors for convolution
    Input = Input.reshape(len(Input),)
    Shaper = Shaper.reshape(len(Shaper),)
    ShInput = np.convolve(Input,Shaper)

    # Delete convolution remainder
    ShapedInput=ShInput[0:rows+shlen-1]

    # Define end of command time and round to account for numerical errors
    end_time = (len(ShapedInput))*deltaT

    end_time = np.round(end_time, int(np.abs(np.log10(deltaT))))

    # Create the "shaped" time vector to output
    T= np.arange(0, end_time, deltaT)

    error = len(T) - len(ShapedInput)

    if error > 0:
      ShapedInput = np.append(ShapedInput,np.zeros(error))


    # return the "shaped" time vector and the shaped input
    return T, ShapedInput


def sensplot(seq, fmin, fmax, zeta, numpoints = 2000, plotflag = 0):
    """Original MATLAB preamble
        sensplot  Plot the residual over range of frequencies

        list = sensplot(seq,fmin,fmax,zeta,numpoints,plotflag)

        seq is the shaping sequence
        fmin is the low end of the frequency range
        fmax is the high end of the frequency range
        zeta is the damping ratio of the system
        numpoints is the number of points to calculate, default is 2000
        plotflag plots the data if plotflag=1, default is 0


    Converted to Python on 2/26/13 by Joshua Vaughan (joshua.vaughan@louisiana.edu)"""

    fmax = float(fmax) # force one value to be floating point, to ensure floating point math
    df = (fmax-fmin)/numpoints


    [rows,cols] = np.shape(seq)
    tn = seq[-1,0]
    frequency = np.zeros((numpoints,1))
    amplitude = np.zeros((numpoints,1))

    # the vibration percentage formulation is:
    #  t(i) is seq(i,1)
    #  A(i) is seq(i,2)
    #  tn is seq(num_of_rows_in_seq,1)
    for nn in range(numpoints):
        sintrm = 0
        costrm = 0
        freq = (fmin + nn*df)*2*np.pi

        for i in range(rows):
            sintrm = sintrm + seq[i,1]*np.exp(zeta*freq*seq[i,0])*np.sin(freq*np.sqrt(1-zeta**2)*seq[i,0])
            costrm = costrm + seq[i,1]*np.exp(zeta*freq*seq[i,0])*np.cos(freq*np.sqrt(1-zeta**2)*seq[i,0])

        frequency[nn,0] = freq/2.0/np.pi
        amplitude[nn,0] = np.exp(-zeta*freq*tn)*np.sqrt(sintrm**2+costrm**2)

    if plotflag == 1:
        plt.plot(frequency, amplitude*100)
        plt.xlabel(r'Frequency (Hz)',fontsize=22,weight='bold',labelpad=5)
        plt.ylabel(r'Percentage Vibration',fontsize=22,weight='bold',labelpad=8)
        plt.show()

    return frequency, amplitude

def seqconv(shaper1, shaper2):
    """ Original MATLAB preamble
   SEQUENCECONVOLVE Convolve two continuous sequences together.

   seq = seqconv(seq1,seq2)

   Convolves two sequences together.
   A Sequence is an n*2 matrix with impulse times (sec) in
   the first column and amplitudes in the second column.

   Parameters:
    shaper1, shaper2 the two sequences to convolve together.

   Returns:
    seq, the sequence resulting from the convolution.

   Converted to Python on 01/16/15 by Joshua Vaughan - joshua.vaughan@louisiana.edu
   """

    index = 0
    tempseq = np.zeros((np.shape(shaper1)[0] * np.shape(shaper2)[0], 2))

    for ii in range(len(shaper1)):
        for jj in range(len(shaper2)):
            tempseq[index, 0] = shaper1[ii, 0] + shaper2[jj, 0]
            tempseq[index, 1] = shaper1[ii, 1] * shaper2[jj, 1]
            index += 1

    num_impulses = index
    newshaper = np.asarray(seqsort(tempseq))
    
    times = []
    amps = []
    
    idx = np.argsort(newshaper[:,0])
    shap_sort = newshaper[idx, :]

    tms_X0 = shap_sort[:,0]
    amps_X0 = shap_sort[:,1]
    # check for impulses occuring at the same time 
    # if so, remove one, sum amplitudes, and resolve
    
    times = np.append(times, tms_X0[0])
    amps = np.append(amps, amps_X0[0])

    
    for ii in range(1,newshaper.shape[0]):
        #pdb.set_trace()
        if np.abs(tms_X0[ii] - tms_X0[ii-1]) < 1e-4:
            #print('\nRepeated Times. Shortening and resolving...')
            amps[-1] = amps[-1] + amps_X0[ii]
        else:

            times = np.append(times, tms_X0[ii])
            amps = np.append(amps, amps_X0[ii])
    
    # create new initial Guess
    X0 = np.hstack((amps, times))

    # Put the result in standard shaper form
    num_impulses = len(amps)
    amps = amps.reshape(num_impulses,1)
    times = times.reshape(num_impulses,1)

    shaper = np.hstack((times,amps))

    return shaper


def seqsort(shaper_sequence):
    """ Function to sort a shaper sequence

    Used mainly in the solution of two-mode shapers. Following convolution
    these shapers are often mis-ordered or have multiple impulses at
    identical times. This function sorts the impulses according to time, then
    attemptes to resolve any multi-impulse time locations.

    Arguments:
      shaper_sequence : A typical [ti Ai] Nx2 shaper array

    Returns:
      sorted : The properly sorted and possibly shortened version
               of shaper_sequence

    Created: 01/16/15 - Joshua Vaughan - joshua.vaughan@louisiana.edu
    """

    # Sort the sequence according to the impulse times (in the first column)
    time_sorted = shaper_sequence[shaper_sequence[:,0].argsort()]

#     print time_sorted

    # Check if the number of unique times is equal to the number of rows
    if len(np.unique(time_sorted[:,0])) != len(time_sorted[:,0]):
        # If the lengths are not equal, there is a repeated time.
        # Find it and combine the impulse amplitudes

#         import pdb; pdb.set_trace()
        shortened = np.zeros((len(np.unique(time_sorted[:,0])), 2))

#         print '\nFinal length should be: ' + str((len(np.unique(time_sorted[:,0]))))

        row = 0
        ii = 0

        index = 0
        for time, amp in time_sorted:
#             print '\nindex: {}'.format(index)
#             print 'Current time {}'.format(time)
#             print 'Current amp {}'.format(amp)
#             print 'Current seq:\n{}'.format(shortened)
#
            if time in shortened[:, 0]:
                repeating_row = np.where(time==shortened[:, 0])[0][0]
#                 print 'Adding on row {}'.format(repeating_row)
                shortened[repeating_row, 1] = shortened[repeating_row, 1] + amp
#                 print 'After adding {}'.format(shortened)
                if time == 0:
                    index += 1
            else:
#                 print 'Non-repeated time'
                shortened[index, :] = np.array([time, amp])
#                 print 'Resulting seq:\n{}'.format(shortened)
                index += 1

        sorted = shortened

    else:
        sorted = time_sorted

    return sorted


def bang_bang(CurrTime, Amax, Vmax, Distance, StartTime = 0.0):
    """
    Function to create a bang-bang or bang-coast-bang acceleration command

    Arguments:
      CurrTime : The current timestep (or an array of times)
      Amax : maximum acceleration of the command
      Vmax : maximum velocity of the resulting command
      Distance : How far the system would move from this command
      StartTime : When the command should begin

    Returns :
      The acceleration commmand for the current timestep CurrTime or if an
      array of times was passed, the array representing the input over that
      time period.
    """
    #sign = np.abs(Distance) / Distance

    Distance = np.round(Distance,2)

    # These are the times for a bang-coast-bang input
    t1 = StartTime
    t2 = np.round(Vmax/Amax,10) + t1
    t3 = np.round(np.abs(Distance)/Vmax,10) + t1
    t4 = (t2 + t3)-t1
    end_time = t4

    if Distance < 0.: 
      Amax *= -1

    if t3 <= t2: # command should be bang-bang, not bang-coast-bang
        t2 = np.sqrt(np.round((np.abs(Distance)/Amax),10))+t1
        t3 = 2.0 * np.sqrt(np.round((np.abs(Distance)/Amax),10))+t1
        end_time = t3

        accel = Amax*(CurrTime > t1) - 2*Amax*(CurrTime > t2) + Amax*(CurrTime > t3)

    else: # command is bang-coast-bang
        accel = Amax*(CurrTime > t1) - Amax*(CurrTime > t2) - Amax*(CurrTime > t3) + Amax*(CurrTime > t4)
    
    return accel


def pulse(CurrTime, force, duration, StartTime = 0.0,impulse_sign=1):
    """
    Function to create a bang-bang or bang-coast-bang acceleration command

    Arguments:
      CurrTime : The current timestep (or an array of times)
      Amax : maximum acceleration of the command
      Vmax : maximum velocity of the resulting command
      Distance : How far the system would move from this command
      StartTime : When the command should begin

    Returns :
      The acceleration commmand for the current timestep CurrTime or if an
      array of times was passed, the array representing the input over that
      time period.
    """

    # These are the times for a bang-coast-bang input
    t1 = StartTime
    t2 = duration + t1

    accel = impulse_sign*force*(CurrTime > t1) - impulse_sign*force*(CurrTime > t2) 

    return accel



def step_input(CurrTime, Amp, StartTime = 0.0):
    """
    Function to create a step input

    Arguments:
      CurrTime : The current timestep (will also take an array)
      Amp : The size of the step input
      StartTime : The time that the step should occur

    Returns:
      The step input for the CurrTime timestep or an array representing the
      step input over the times pass
    """

    return Amp * (CurrTime > StartTime)


def s_curve(CurrTime, Amp, RiseTime, StartTime=0.0):
    """
    Function to generate an s-curve command

    Arguments:
      CurrTime : The current timestep or an array of times
      Amp : The magnitude of the s-curve (or final setpoint)
      RiseTime : The rise time of the curve
      StartTime : The time that the command should StartTime

    Returns :
      The command at the current timestep or an array representing the command
      over the times given (if CurrTime was an array)
    """

    scurve = 2.0 * ((CurrTime - StartTime)/RiseTime)**2 * (CurrTime-StartTime >= 0) * (CurrTime-StartTime < RiseTime/2) \
            +(-2.0 * ((CurrTime - StartTime)/RiseTime)**2 + 4.0 * ((CurrTime - StartTime)/RiseTime) - 1.0) * (CurrTime-StartTime >= RiseTime/2) * (CurrTime-StartTime < RiseTime) \
            + 1.0 * (CurrTime-StartTime >= RiseTime)

    return Amp * scurve

def trapezoidal_command(CurrTime, Distance, Vmax, Accel, StartTime=0.):
    """
    Function to generate a trapezoidal velocity command
    
    Arguments:
      CurrTime : The current timestep or an array of times
      Distance : The distance to travel over
      Vmax: The maximum velocity to reach
      Accel: The acceleration, assumed to be symmetric
      StartTime : The time that the command should StartTime
      
    Returns :
      The command at the current timestep or an array representing the command
      over the times given (if CurrTime was an array)
    """
    
    t1 = StartTime + Vmax / Accel
    t2 = Distance / Vmax + StartTime
    t3 = t2 + Vmax / Accel
    
    #pdb.set_trace()

    # We'll create the command by just superimposing 4 ramps starting at 
    # StartTime, t1, t2, and t3
    trapezoidal = (Accel * (CurrTime - StartTime) * (CurrTime - StartTime >= 0) +
                   -(Accel * (CurrTime - t1) * (CurrTime - t1 >= 0)) +
                   -(Accel * (CurrTime - t2) * (CurrTime - t2 >= 0)) +
                   (Accel * (CurrTime - t3) * (CurrTime - t3 >= 0)))
    
    return trapezoidal

def ramp(CurrTime,Distance,Vmax,StartTime=0.):

  t1 = StartTime
  t2 = Distance / Vmax + t1

  ramp = Vmax * (CurrTime - StartTime) * (CurrTime > t1) - Vmax * (CurrTime - StartTime) * (CurrTime > t2)

  return ramp


def shaped_input(unshaped_func, CurrTime, Shaper, *args):
    """
    Function to create a shaped input given a function for an unshaped command

    Arguments:
      unshaped_func : function representing the unshaped command
                      must accept current timestep as its first argument
      CurrTime : The current timestep to deteremint the input for
      Shaper : The shaper to use, should be in [ti Ai] form
      *args : optional arguments to pass to unshaped_func

    Returns:
      shaped : the current timestep of the shaped command
    """

    shaped = 0.0

    for impulse_time, impulse_amp in Shaper:
        shaped = shaped + impulse_amp * unshaped_func(CurrTime - impulse_time, *args)

    return shaped

def impulse(CurrTime, Amax, Vmax, StartTime = 0.0,impulse_sign=1):
    """
    Function to create a bang-bang or bang-coast-bang acceleration command

    Arguments:
      CurrTime : The current timestep (or an array of times)
      Amax : maximum acceleration of the command
      Vmax : maximum velocity of the resulting command
      Distance : How far the system would move from this command
      StartTime : When the command should begin

    Returns :
      The acceleration commmand for the current timestep CurrTime or if an
      array of times was passed, the array representing the input over that
      time period.
    """

    # These are the times for a bang-coast-bang input
    t1 = StartTime
    t2 = np.round(Vmax/Amax,10) + t1

    #print(impulse_sign)

    accel = impulse_sign*Amax*(CurrTime > t1) - impulse_sign*Amax*(CurrTime > t2) 

    return accel

def CRAWLAB_fft(data, time, freq_max):
    ''' Function to get the FFT for a response
    #
    # Inputs:
    #   time = time array corresponding to the data
    #   data = the response data array (only pass a single dimension/state at at time)
    #   plotflag = will plot the FFT if nonzero
    #   
    # Output:
    #   fft_freq = an array of the freqs used in the FFT
    #   fft_mag = an array of the amplitude of the FFT at each freq in fft_freq
    #
    # Created: 03/28/14
    #   - Joshua Vaughan
    #   - joshua.vaughan@louisiana.edu
    #   - http://www.ucs.louisiana.edu/~jev9637
    #
    # Modified:
    #   * 03/17/16 - JEV - joshua.vaughan@louisiana.edu
    #       - updated for Python 3
    ######################################################################################
    '''
    
    from scipy.fftpack import fft

    # correct for any DC offset
    offset = np.mean(data) 

    # Get the sampling time
    sample_time = time[1] - time[0]
    
    # Get the length of the dataset
    n = len(data)

    # Calculate the FFT of the data, removing the offset and using a Hanning Window
    fft_mag = fft((data - offset))
    
    # Define the frequency range of the output
    freq = np.linspace(0.0, 1.0 / (2.0*sample_time), int(np.ceil(n/2)))
    
    # Only return the "useful" part of the fft
    mag = 2.0/n * np.abs(fft_mag[0:int(np.ceil(n/2))])

    return freq,mag

