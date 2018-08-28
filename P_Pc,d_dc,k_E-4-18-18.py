#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:22:03 2018

@author: SRattan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:50:23 2016

@author: SRattan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 21:10:16 2016
This program plots the data from the contact point to the puncture point, and the fitting with equation a(x-x_o)^2 + b(x-x_o)
as well the residuals for each run. At the end, you will see the avergae k'E with stdev for all the runs.
conclusion is : no matter where you click on the flat region of the force-displacement curve, the program automatically detects the 
point where the force starts building up. It is able to find x_o by itself if you provide an initial guess (by clicking).
@author: SRattan
"""


import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import lmfit
import glob,os
from lmfit import Parameters, minimize, report_fit
import numpy as np
import pylab as pl
from scipy.optimize import curve_fit
from pylab import ginput
from termcolor import colored
from matplotlib.font_manager import FontProperties
from numpy import trapz
import matplotlib.gridspec as gridspec
from scipy import stats
mpl.rc('font',family='Times New Roman')
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind
def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

def residual_1(params, x, data):        # residuals for equation a(x)^2,objective function that has to be minimized 
    a = params['quad_term'].value
    model_1 = a*x**2
    return model_1-data
    

def residual_2(params, x, data):        # residuals for whole equation a(x-x_o)^2 + b(x-x_o),objective function that has to be minimized 
    a = params['quad_term'].value
    b = params['linear_term'].value
    x_o = params['xoffset'].value
    model_2_appen=[]
    for i in range(len(x)):
      if x[i] >= x_o:
       model_2 = a*(x[i]-x_o)**2 + b*(x[i]-x_o) 
      else:
       model_2 = (x_o)*0
      model_2_appen.append(model_2)
    return model_2_appen-data
       
    
def fitting1(x,a):
    return a*x**2
    

def fitting2(x,a,b,x_o):
     y_fit_appen = []
     for i in range(len(x)):     
       if x[i] < x_o:
           y_fit = 0
       else:
           y_fit = (a*(x[i]-x_o)**2 + b*(x[i]-x_o))
       y_fit_appen.append(y_fit)    
     return y_fit_appen
def smooth(y, radius):  
    y_smoothed = []
    for i in range(len(y)):
        avg=0
        if i > radius and i < len(y)-radius: 
            for j in range(i-radius,i+radius): 
                avg+=y[j]
            avg=avg/(2*radius) 
        elif i>=len(y)-radius:
             avg = y[i]
        y_smoothed.append(avg) 
    return y_smoothed
      
Pc_arr =[]
dc_arr1=[]
dc_arr2=[]
dc_sum_1 = 0
dc_sum_2 = 0
Pc_sum = 0
count = 0
C=[]
force_avg=[] #stores avergae force, f_avg computed from 1000 datapoints in front of the contact point
E =[]
U_gel=[]
U_gel_sum=0
U_cant=[]
U_cant_sum=0
E_sum = 0
lt=[]
lt_sum=0
count = 0
f_zero_signal=[]
print ('select curve fitting end point first(very close to zero force), then contact point and peak last')
path = raw_input("Enter foldername:")
v=10 
#v=raw_input('Enter velocity of the test in um/s:')
leg  = raw_input("Enter legend for these files:")
os.chdir(path)
stiff = raw_input("Enter cantilever stiffness:")
listindir = os.listdir(path)
os.chdir(path)
for j in listindir:
    if j.endswith(".txt"): #and j.startswith("run"):
            peaks_idx=[]
            font = FontProperties()
            count += 1
            ct = float(j[3])
            cnt=ct+1
            file = np.loadtxt(j, skiprows=1)
            d = file[:,1]
            f_raw= -file[:,2]
            t = file[:,0]
            f_list = smooth(f_raw,7)       # f is the smoothed force array
            f=np.asarray(f_list)           # smooth spits out list, conver to array
            no_datapoints_baseline = 2000 
            gel_d = d - (f/float(stiff))
            #cant_d = f/float(stiff)
            figct = plt.figure(figsize=(15,8))
            plt.scatter(t,f,label=j)
            #plt.scatter(t,f,label=j)
            plt.legend(loc='best')
            pl.xlabel("time(s)")
            #pl.xlabel("Gel displacement(um)")
            pl.ylabel("load(mN)")
            #figct,axarr = plt.subplots(2,figsize=(15,8))
            #gs = gridspec.GridSpec(2,2)
            #axarr[0] = plt.subplot(gs[0, :])
            #axarr[0].plot(gel_d,f,label=j)         #in case you want a subplot of force vs gel displacement
            #pl.xlabel("Gel displacement(um)")
            #pl.ylabel("load(mN)")
            max_disp = max(gel_d)
            max_idx_d = None
            for i in range(len(gel_d)):
             if max_idx_d is None and gel_d[i] == max_disp:
                max_idx_d= i
                break
            assert max_idx_d is not None  
            #axarr[1] = plt.subplot(gs[1,:])       #in case you want a subplot of force vs cantilever displacement
            #axarr[1].plot(cant_d,f,label=j)
            #plt.legend(loc='upper left')
            #pl.xlabel("cant displacement(um)")
            #pl.ylabel("load(mN)")
            #pl.ylim(ymin=-0.5,ymax=3)
            #pl.xlim(xmin=300)
            pts = ginput(3, timeout=0)
            ep_x = pts[0][0]*float(v)      #[first point][x-coordinate] click first. point as close as possible to contact point. For accurate contact point selection
            ep_y = pts[0][1]      #[first point][y-coordinate]
            cp_x = pts[1][0]*float(v)      #click second. approximate, manually picked contact point
            pk_y = pts[2][1]       #clicked third. this is close to peak puncture force
            pk_x = pts[2][0] *float(v)
            #print('Manual peak puncture force in mN is', pk_y)
            #print('Manual peak puncture depth in uN is', pk_x-cp_x)
            max_f= max(f)
            start_idx_f = None
            for i in range(len(f)):
              if start_idx_f is None and f[i]>=0.08*max_f: #start_idx_f sets the section of the data where you need detectpeaks for picking peak puncture force(Pc) to work. you want to set this region just around Pc.
                start_idx_f= i
                break
            assert start_idx_f is not None 
            start_idx = None  #contact point
            end_idx = None    #last point of the section to be fitted for accurate contact point picking
            for i in range(len(gel_d)):
                if start_idx is None and gel_d[i] > cp_x:
                    start_idx = i
                if end_idx is None and gel_d[i] > ep_x:
                    end_idx   = i
                    break
            assert start_idx is not None
            assert end_idx is not None
            idx_1 = int(start_idx-no_datapoints_baseline*1.1)
            idx_2 = int(start_idx-no_datapoints_baseline*0.1)
            f_zero_signal= f[np.maximum(idx_1,1): idx_2 ] #idx_1 to idx_2 is 100 to 1100 indices before contact point
            f_avg = np.mean(f_zero_signal)
            force_avg.append(f_avg)
            f_new=f[start_idx_f:max_idx_d]-f_avg # for finding peaks only in the region around puncture
            gel_d_1= gel_d[start_idx:end_idx]-gel_d[start_idx]
            f_1= f[start_idx:end_idx]-f_avg
            coeff1 = curve_fit(fitting1, gel_d_1,f_1)
            a = coeff1[0][0]    
            #print ('kprimeE from just fitting1 in kPa is', a*10**6)  #This 'a' provides an input for quadratic term of fitting 2
            params_1 = Parameters()
            params_1.add('quad_term', value = a, vary=True)
            result = lmfit.minimize(residual_1, params_1, args=(gel_d_1, f_1)) #minimizes residuals to obtain optimum paramters
            final_1 = f_1 + residual_1(result.params, gel_d_1,f_1)  # calculate final result
            #print 'Report for model 1 is',result.params
            q = result.params.valuesdict() #print 'best fit for quad_term for fitting1 is:',q['quad_term'],':',j
            ct +=1 
            # try to plot results
            try:
               figct= plt.figure(figsize=(15,8))
               frame1 = figct.add_axes((.1,.3,.8,.6), label='Load(mN)')
               plt.text(0,2,a)
               line_1, = plt.plot(gel_d_1, f_1, '.b')
               line_2, = plt.plot(gel_d_1, final_1, 'r')
               plt.legend([line_1,line_2], ['Loading curve of'+j,'fitting 1: ax^2'])
               plt.show()
               #plt.legend(loc='best')
               frame1.set_xticklabels([])
               frame1.set_ylabel('Load(mN)')
               frame2=figct.add_axes((.1,.1,.8,.2), label='Residuals')  
               frame2.set_ylabel('Residuals')
               frame2.set_xlabel('Displacement(um)')
               plt.plot(gel_d_1,residual_1(result.params, gel_d_1,f_1),'or')
               time.sleep(1)
            except:
               pass
            
            f_2 = f[:end_idx]-f_avg  #f_avg subtraction critical for finding contact point
            gel_d_2 = gel_d[:end_idx]
            start = gel_d[0]
            #cant_d_1=cant_d[start_idx:end_idx]
            coeff2 = curve_fit(fitting2, gel_d_2, f_2)
            b_1=coeff2[0][1]
            params_2 = Parameters()
            params_2.add('quad_term', value = q['quad_term'], vary= True)
            params_2.add('linear_term',value= -0.000000011, vary=True)
            params_2.add('xoffset', value = cp_x, vary= True)
            result_2 = lmfit.minimize(residual_2, params_2, args=(gel_d_2, f_2))
            #print 'Report for model 2 and run',ct-1,'is:', '\n'
            m = result_2.params.valuesdict()
            #modulus = m['quad_term']*10**6
            d_o= m['xoffset']
            l_t_1=m['linear_term']
            #print('cp_x is', cp_x,'um &','d_o', d_o,'um')
            final_2 = f_2 + residual_2(result_2.params, gel_d_2,f_2)
            ct +=1 
            #f_1_smooth= smooth(f_1,5)
            #area1 = trapz(f_1,gel_d_1)
            #U_gel.append(area1)
            #U_gel_sum+=area1    #for N/m
            #area2 = trapz(f_1,cant_d_1)
            #U_cant.append(area2)
            #U_cant_sum+=area2   
            #print('U_E,gel:',area1,'mN-um','for',j)
            #print('U_E,cant',area2,'mN-um')
            figct = plt.figure(figsize=(15,12))
            frame1 = figct.add_axes((.1,.35,.8,.6), label='Load(mN)')
            frame1.set_ylabel('Load(mN)')
            plt.text(0,2,m)
            line_1,= plt.plot(gel_d_2, f_2, '.b')
            line_2,=plt.plot(gel_d_2, final_2, 'r')
            plt.legend([line_1,line_2], ['Loading curve upto small force '+j,'fitting 2: a(x-xo)^2+b(x-xo)'])
            plt.show()
            #line_1.set_legend(loc='best', framealpha=.5, numpoints=1)
            #plt.legend(loc='best')
            frame1.set_xticklabels([])
            frame2=figct.add_axes((.1,.15,.8,.2), label='Residuals')     
            frame2.set_ylabel('Residuals')
            frame2.set_xlabel('Displacement(um)')
            plt.plot(gel_d_2,residual_2(result_2.params, gel_d_2,f_2),'or')
            plt.grid()
            peakind= detect_peaks(f_new,valley=False,mpd=400,show=True,mph=(pk_y)*0.9, threshold=0) #this code line tries to find peaks in the section of data around Pc.I set mph as very close to manual peak puncture force(pk_y) so that it most likely picks the first peak as Pc. It can then pick several closely spaced (set by mpd) peaks after puncture which doesn't matter much as long as peakind[0][0] is Pc which is is true for 90% cases  
            #mph : {None, number}, optional (default = None)
            #detect peaks that are greater than minimum peak height.
            #mpd : positive integer, optional (default = 1)
            #detect peaks that are at least separated by minimum peak distance (in
            #number of data).
            #threshold : positive number, optional (default = 0)
            #detect peaks (valleys) that are greater (smaller) than `threshold`
            #in relation to their immediate neighbors.
            #peakind= detect_peaks(f_new,valley=True,mpd=10,show=True)#: 1mm/sec
            peaks_idx.append(peakind) 
            fv_idx = peaks_idx[0][0]+ start_idx_f #indext of peak puncture force
            contact_idx_d_o = None
            for i in range(len(gel_d)):
               if contact_idx_d_o is None and gel_d[i]> d_o:
                contact_idx_d_o = i-1
                break
            assert contact_idx_d_o is not None
            Pc = f_raw[fv_idx]  # correct Pc. Algorithm calculated the contact point and peak punctrue force
            force = f[contact_idx_d_o:fv_idx]# this data is a the corrected force for zero displacement until the true Pc
            #print ('peak puncture force in mN for',j,'is', Pc)
            Pc_arr.append(Pc-f_avg) #for standard deviation
            Pc_sum+=(Pc-f_avg)
            dc_1= gel_d[fv_idx]-d_o   #subtract d_o (or x_offset) obtained from algorithm from peak puncture depth 
            dc_2 = gel_d[fv_idx]-cp_x  #subtract x_offset or cp_x manually picked up from peak puncture depth 
            dc_arr1.append(dc_1)
            dc_arr2.append(dc_2)
            dc_sum_1+=dc_1
            dc_sum_2+=dc_2
            print ('for',j,Pc,'mN', dc_1,'(from algorithm) um')
            gel_d_n = (gel_d[contact_idx_d_o:fv_idx]-gel_d[contact_idx_d_o])/dc_2 # normalized gel_d_n based on manual contact point
            #peak_idx_pc = None
            #for i in range(len(force)):
              #if peak_idx_pc is None and force[i]>= Pc:
                #peak_idx_pc = i    #find index of Pc
                #break
            #assert peak_idx_pc is not None
            #print('Pc from fv_idx is',Pc,'Pc from peak_idx_pc is',force[peak_idx_pc])
            f_final_n = (f[contact_idx_d_o:fv_idx]-f_avg)/Pc
            #dc_idx = None
            #for i in range(len(f)):
               #if dc_idx is None and f[i] >= Pc:   # find idx of dc
                 #dc_idx = i
                 #print dc_idx
                 #break
              #assert dc_idx is not None  
            
              #print ('f_final_n is',f_final_n)
              #print('peak puncture depth,dc is', dc)
              #print('gel_d[dc_idx]-gel_d[contact_idx_d_o] is:',gel_d[dc_idx]-gel_d[contact_idx_d_o])
              #print('d_o is:',d_o)
              #print('gel_d[contact_idx_d_o] is:',gel_d[contact_idx_d_o])
              #print('Length is',len(peaks_idx))
            #fv_idx=None 
            #if fv_idx is None and len(peaks_idx)==1:
            #print ( 'I entered if')
              #fv_idx = max_idx_d
            #print('fv_idx is', fv_idx)
            #elif fv_idx is None and len(peaks_idx)>1:
              #fv_idx = peaks_idx[0][1]+ start_idx_f
            #assert fv_idx is not None
            #print('fv_idx is', fv_idx)
            gel_d_final= (gel_d[contact_idx_d_o:fv_idx-1]-gel_d[contact_idx_d_o])
            f_final= (f_raw[contact_idx_d_o:fv_idx-1]-f_avg)
            coeff1 = curve_fit(fitting1, gel_d_final,f_final)
            a_1 = coeff1[0][0]    
            print ('kprimeE (a_1) from just fitting1 in kPa is', a_1*10**6)
            params_3 = Parameters()
            params_3.add('quad_term', value = a_1, vary=True)
            result = lmfit.minimize(residual_1, params_3, args=(gel_d_final, f_final)) #minimizes residuals to obtain optimum paramters
            final = f_final + residual_1(result.params, gel_d_final,f_final)  # calculate final result
              #print 'Report for model 1 is',result.params
            q_1 = result.params.valuesdict() #print 'best fit for quad_term for fitting1 is:',q['quad_term'],':',j
            f_2 = f_raw[:fv_idx]-f_avg  #f_avg subtraction critical for finding contact point
            gel_d_2 = gel_d[:fv_idx]
            #cant_d_1=cant_d[start_idx:end_idx]
            coeff2 = curve_fit(fitting2, gel_d_2, f_2)
            b_2=coeff2[0][1]
            params_4 = Parameters()
            params_4.add('quad_term', value = q_1['quad_term'], vary= True)
            #params_4.add('linear_term',value=l_t_1,vary=False)
            params_4.add('linear_term',value= 0.000000081, vary=True)
            params_4.add('xoffset', value = d_o, vary= False)
            result_4 = lmfit.minimize(residual_2, params_4, args=(gel_d_2, f_2))
              #print 'Report for model 1 is',result.params
            #print 'Report for model 2 and run',ct-1,'is:', '\n'
            #result_2.params.pretty_print()
            #print_params=result_4.params.pretty_print()
            m_1 = result_4.params.valuesdict()
            final_4 = f_2 + residual_2(result_4.params, gel_d_2,f_2)  # calculate final result
            modulus = m_1['quad_term']*10**6
            linear_trm=m_1['linear_term']*10**6
            lt.append(linear_trm)
            lt_sum+=linear_trm
            #print('d_o', d_o)
            print ('kprimeE from full curve for',j,'is',modulus,'kPa')
            E. append(modulus)
            E_sum += modulus
              #final_2 = f_2 + residual_2(result_2.params, gel_d_2,f_2)
             
              #print ('f_final is',f_final)
              #gel_d_list= smooth(gel_d,7)
              #f_final_list=smooth(f_final,7)
              #f_final=np.asarray(f_final_list) 
              #gel_d=np.asarray(gel_d_list) 
            ct +=1
            figct = plt.figure(figsize=(15,12))
            frame1 = figct.add_axes((.1,.35,.8,.6), label='Load(mN)')
            frame1.set_ylabel('Load(mN)')
            line_1,= plt.plot(gel_d_2, f_2, '.b')
            line_2,=plt.plot(gel_d_2, final_4, 'r')
            plt.text(0,2,m_1)
            plt.legend([line_1,line_2], ['Loading FULL curve of '+j,'fitting 2: a(x-xo)^2+b(x-xo)'])
            #line_1.set_legend('Loading curve of FULL curve'+j,'fitting 2: a(x-xo)^2+b(x-xo)')
            plt.show()
            #plt.legend(loc='best')
            frame1.set_xticklabels([])
            frame2=figct.add_axes((.1,.15,.8,.2), label='Residuals')     
            frame2.set_ylabel('Residuals')
            frame2.set_xlabel('Displacement(um)')
            plt.plot(gel_d_2,residual_2(result_4.params, gel_d_2,f_2),'or')
            plt.grid()
            if j[3] is 'P': 
               fig0 = plt.figure(0,figsize=(6.693,6.693),linewidth=3,tight_layout=True) 
               ax1= fig0.add_subplot(111)
               ax1.set_title(r'$d_{c}$ is based on x-offset determined by algorithm')
               #ax2= fig0.add_subplot(212)
               linect= ax1.plot(gel_d_final,f_final,'r',label=j+'-'+leg) #+r'$\phi_{v}=$')#'$\mu m$') 
               #linecnt = ax2.plot(gel_d_n,f_final_n,'k',label=j+'-'+leg)
               plt.legend(loc='best',fontsize=15) 
               ax1.minorticks_on()
               #ax2.minorticks_on()
               for axis in ['top','bottom','left','right']:
                 ax1.spines[axis].set_linewidth(3)
                 #ax2.spines[axis].set_linewidth(3)
               ax1.tick_params(which='both',bottom='on',top='on', left='on', right='on')
               ax1.tick_params(axis='x',which='major',direction='in',length=8,width=3,labelsize=15)#,fontname="Times New Roman",fontsize=16)
               ax1.tick_params(axis='x',which='minor',direction='in',length=4,width=3,labelsize=15)#,fontname="Times New Roman",fontsize=16)
               ax1.tick_params(axis='y',which='major',direction='in',length=8,width=3,labelsize=15)#,fontname="Times New Roman",fontsize=16)
               ax1.tick_params(axis='y',which='minor',direction='in',length=4,width=3,labelsize=15)#,fontname="Times New Roman",fontsize=16)
               ax1.set_xlabel(r'$d (\mu m)$',fontsize=15)
               ax1.set_ylabel(r'$P (mN)$',fontsize=11.5) 
               #ax2.tick_params(which='both',bottom='on',top='on', left='on', right='on')
               #ax2.tick_params(axis='x',which='major',direction='in',length=8,width=3,labelsize=15)#,fontname="Times New Roman",fontsize=16)
               #ax2.tick_params(axis='x',which='minor',direction='in',length=4,width=3,labelsize=15)#,fontname="Times New Roman",fontsize=16)
               #ax2.tick_params(axis='y',which='major',direction='in',length=8,width=3,labelsize=15)#,fontname="Times New Roman",fontsize=16)
               #ax2.tick_params(axis='y',which='minor',direction='in',length=4,width=3,labelsize=15)#,fontname="Times New Roman",fontsize=16)
               #ax2.set_xlabel(r'$d/d_{c}$',fontsize=15)
               #ax2.set_ylabel(r'$P/P_{c}$',fontsize=15) 
               #ax2.set_ylim(ymax=1.2)
               #ax2.set_xlim(xmax=1.2)
            else:
             fig1 = plt.figure(0,figsize=(6.693,6.693),linewidth=3) 
             ax1= fig1.add_subplot(111)
            #ax2= fig0.add_subplot(212)
             linect= ax1.plot(gel_d_final,f_final,'g',label=j+'-'+leg) 
            #linecnt = ax2.plot(gel_d_n,f_final_n,'k',label=j+'-'+leg)
             ax1.minorticks_on()
             #ax2.minorticks_on()
             plt.legend(loc='best',fontsize=11.5) 
             for axis in ['top','bottom','left','right']:
                 ax1.spines[axis].set_linewidth(3)
                 #ax2.spines[axis].set_linewidth(3)
             ax1.tick_params(which='both',bottom='on',top='on', left='on', right='on')
             ax1.tick_params(which='both',bottom='on',top='on', left='on', right='on')
             ax1.tick_params(axis='x',which='major',direction='in',length=8,width=3,labelsize=15) #fontname="Times New Roman",fontsize=16)
             ax1.tick_params(axis='x',which='minor',direction='in',length=4,width=3,labelsize=15) #fontname="Times New Roman",fontsize=16)
             ax1.tick_params(axis='y',which='major',direction='in',length=8,width=3,labelsize=15) #fontname="Times New Roman",fontsize=16)
             ax1.tick_params(axis='y',which='minor',direction='in',length=4,width=3,labelsize=15) #fontname="Times New Roman",fontsize=16) 
             ax1.set_xlabel(r'$d (\mu m)$',fontsize=15)
             ax1.set_ylabel(r'$P (mN)$',fontsize=15) 
            #ax2.tick_params(which='both',bottom='on',top='on', left='on', right='on')
            #ax2.tick_params(which='both',bottom='on',top='on', left='on', right='on')
            #ax2.tick_params(axis='x',which='major',direction='in',length=8,width=3,labelsize=15) #fontname="Times New Roman",fontsize=16)
            #ax2.tick_params(axis='x',which='minor',direction='in',length=4,width=3,labelsize=15) #fontname="Times New Roman",fontsize=16)
            #ax2.tick_params(axis='y',which='major',direction='in',length=8,width=3,labelsize=15) #fontname="Times New Roman",fontsize=16)
            #ax2.tick_params(axis='y',which='minor',direction='in',length=4,width=3,labelsize=15) #fontname="Times New Roman",fontsize=16) 
            #ax2.set_xlabel(r'$d/d_{c}$',fontsize=15)
            #ax2.set_ylabel(r'$P/P_{c}$',fontsize=15) 
            #ax2.set_ylim(ymax=1.2)
            #ax2.set_xlim(xmax=1.2)
              #fig0 = plt.figure(0,figsize=(6.693,6.693))  
              #ax1= fig0.add_axes()
              #linect= ax1.plot(gel_d,f_final,'lime') 
              #ax1.minorticks_on([0.05,0.1,0.5,0.85,1])
              #ax1.tick_params(axis='x',which='major',direction='out',length=4,width=4,color='b',pad=10,labelsize=20,labelcolor='g')
              #plt.legend(loc='best') 
              #ax1.minorticks_on()
              #figct= plt.figure(figsize=(15,8))
              #print (float(j[3]))
               #frame3=figct.add_axes((.1,.15,.8,.2), label='Cantilever f-d') 
               #plt.plot(cant_d_1,f_1)
               #plt.plot(cant_d_1,f_1)
              #delta_f = np.gradient(final_2)
              #gel_d_3= gel_d_2- gel_d[start_idx]
              #max_d = abs(round(np.amax(gel_d_1)))
              #min_d= round(min(gel_d_1))
              #gel_new = np.linspace(min_d,max_d)  
              #C[0:-1]= np.diff(gel_d_2)/np.diff(final_2)
              #C[-1] = (gel_d_2[-1] - gel_d_2[-2])/(final_2[-1] - final_2[-2])
              #C.append(C[-1])
              #print(len(C))
              #print(len(gel_d_2))
              #C = np.gradient(gel_d_2,delta_f)  #to plot Cgel vs gel displacement
              #figct= plt.figure(figsize=(15,8))
              #print('I made compliance figure')
              #plt.scatter(gel_d_2,C)
              #pl.ylim(ymin=-10E-3,ymax=10E+4)
              #pl.xlim(xmin=0)
              #pl.xlabel(r'$Gel\:displacement\:(\mu m)$',fontweight='bold', fontsize=24)
              #pl.ylabel(r'${C}_{gel}\:(\mu m / mN)$',fontsize=24)
              #pl.yscale('log')
              #pl.title(r'${C}_{gel}\: calculation$', fontsize= 30)
            ct+=1 
               #print('I passed')
stdev_Pc=np.std(Pc_arr)
stdev_dc_2=np.std(dc_arr2)
stdev_dc_1=np.std(dc_arr1)
print colored ('Average value of Pc is:','red'),colored (Pc_sum/count, 'red'),colored('±','red'), colored(stdev_Pc,'red'),colored('mN','red')
print colored ('Average value of dc (manual contact pnt) is:','blue'),colored(dc_sum_2/count,'blue'),colored('±','blue'), colored(stdev_dc_2,'blue'), colored('um','blue')
print colored ('Average value of dc (x_offset from algorithm) is:','magenta'),colored(dc_sum_1/count,'magenta'),colored('±','magenta'), colored(stdev_dc_1,'magenta'), colored('um','magenta')
stdev_E=np.std(E)
stdev_lt=np.std(lt)
#stdev_Ugel=np.std(U_gel)
#stdev_Ucant=np.std(U_cant)
print colored ('Average value of kprime is:','green'),colored(E_sum/count,'green'),colored('±','green'), colored(stdev_E,'green')
print ('Average value of lienar_term is:',lt_sum/count,'±',stdev_lt)
print ('Number of runs is',count)
#print colored ('Average value of U_gel in nN-um is:','blue'),colored(U_gel_sum/count,'blue'),colored('pm','blue'), colored(stdev_Ugel,'blue')
#print colored ('Average value of U_cant in mN-um is:','green'),colored(U_cant_sum/count,'green'),colored('pm','green'), colored(stdev_Ucant,'green')'''
            
        