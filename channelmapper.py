import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt, savgol_filter
import scipy.interpolate
from scipy.spatial import distance
from librosa.sequence import dtw # only need the 'dtw' function from the librosa library
import pandas as pd
import shapefile
from shapely.geometry.polygon import LinearRing, orient
from shapely.geometry import Polygon, MultiPolygon, Point, MultiLineString, LineString
from shapely.ops import snap, polygonize, unary_union
from descartes import PolygonPatch
from tqdm.notebook import tqdm, trange
import itertools
import datetime

def resample_and_smooth(x,y,delta_s,smoothing_factor):
    dx = np.diff(x); dy = np.diff(y)      
    ds = np.sqrt(dx**2+dy**2)
    tck, u = scipy.interpolate.splprep([x,y],s=smoothing_factor) # parametric spline representation of curve
    unew = np.linspace(0,1,1+int(sum(ds)/delta_s)) # vector for resampling
    out = scipy.interpolate.splev(unew,tck) # resampling
    xs = out[0]
    ys = out[1]
    return xs, ys

def correlate_curves(x1, x2, y1, y2):
    # use dynamic time warping to correlate two 2D curves
    X = np.vstack((x1,y1))
    Y = np.vstack((x2,y2))
    sm = distance.cdist(X.T, Y.T) # similarity matrix
    D, wp = dtw(C=sm) # dynamic time warping
    p = wp[:,0] # correlation indices for first curve
    q = wp[:,1] # correlation indices for second curve
    return p,q,sm

def compute_curvature(x,y):
    dx = np.gradient(x); dy = np.gradient(y)      
    ds = np.sqrt(dx**2+dy**2)
    ddx = np.gradient(dx); ddy = np.gradient(dy) # second derivatives 
    curvature = (dx*ddy - dy*ddx) / ((dx**2 + dy**2)**1.5)
    s = np.cumsum(ds)
    return curvature, s

def convert_string_to_date(string):
    year = int(string[:4])
    month = int(string[4:6])
    day = int(string[6:])
    date = datetime.datetime(year, month, day)
    return date

def get_migr_rate(x1, x2, y1, y2, years):
    p, q, sm = correlate_curves(x1, x2, y1, y2)
    p = p[::-1] # p and q need to be flipped!
    q = q[::-1]
    qn = np.delete(np.array(q),np.where(np.diff(p)==0)[0]+1)
    pn = np.delete(np.array(p),np.where(np.diff(p)==0)[0]+1)
    xa = x1[:-1]
    xb = x1[1:]
    ya = y1[:-1]
    yb = y1[1:]
    x = x2[qn][1:]
    y = y2[qn][1:]
    migr_sign = np.sign((x-xa) * (yb-ya) - (y-ya) * (xb-xa))
    migr_sign = np.hstack((migr_sign[0], migr_sign))
    migr_dist = migr_sign * sm[pn, qn] / years
    return migr_dist, migr_sign, p, q

def find_zero_crossings(curve, s, x, y):
    n_curv = abs(np.diff(np.sign(curve)))
    n_curv[n_curv==2] = 1
    loc_zero_curv = np.where(n_curv)[0]
    loc_zero_curv = loc_zero_curv +1
    if loc_zero_curv[-1] != len(s)-1:
        loc_zero_curv = np.hstack((0,loc_zero_curv,len(s)-1))
    else:
        loc_zero_curv = np.hstack((0,loc_zero_curv))
    n_infl = len(loc_zero_curv)
    max_curv = np.zeros(n_infl-1)
    loc_max_curv = np.zeros(n_infl-1, dtype=int)
    for i in range(1, n_infl):
        if np.mean(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])>0:
            max_curv[i-1] = np.max(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])
        if np.mean(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])<0:
            max_curv[i-1] = np.min(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])
        max_local_ind = np.where(curve[loc_zero_curv[i-1]:loc_zero_curv[i]]==max_curv[i-1])[0]
        if len(max_local_ind)>1:
            loc_max_curv[i-1] = loc_zero_curv[i-1] + max_local_ind[0]
        elif len(max_local_ind)==1:
            loc_max_curv[i-1] = loc_zero_curv[i-1] + max_local_ind
        else:
            loc_max_curv[i-1] = 0
    # find interpolated zero crossing locations:
    zero_crossings = []
    for i in loc_zero_curv[1:-1]:
        x1 = s[i-1]
        x2 = s[i]
        y1 = curve[i-1]
        y2 = curve[i]
        a = (y2 - y1) / (x2 - x1)
        b = (y1*x2 - y2*x1) / (x2 - x1)
        zero_crossings.append(-b/a)
    zero_x = []
    zero_y = []
    count = 0
    for i in loc_zero_curv[1:-1]:
        x1 = x[i]
        y1 = y[i]
        x2 = x[i+1]
        y2 = y[i+1]
        s1 = s[i]
        s2 = s[i+1]
        s0 = zero_crossings[count]
        x0 = x1 + (x2 - x1)*(s0 - s1)/(s2 - s1)
        y0 = y1 + (y2 - y1)*(s0 - s1)/(s2 - s1)
        zero_x.append(x0)
        zero_y.append(y0)
        count += 1
    return loc_zero_curv, loc_max_curv, zero_crossings, zero_x, zero_y

def create_bars(dates, cutoff_area, dirname, ax):
    # function for creating polygons for 'scroll' bars and plotting them
    bars = [] # these are 'scroll' bars - shapely MultiPolygon objects that correspond to one time step
    erosions = []
    chs = [] # list of channels - shapely Polygon objects
    jumps = [] # gaps between channel polygons that are not cutoffs
    all_chs = [] # list of merged channels (to be used for erosion)
    cutoffs = []
    cmap = mpl.cm.get_cmap('viridis')
    print('create channels and cutoffs...')
    for i in trange(len(dates)-1):
        ch1 = create_channel_polygon_from_shapefile(dirname,dates[i])
        ch2 = create_channel_polygon_from_shapefile(dirname,dates[i+1])
        ch1, bar, erosion, jump, cutoff = one_step_difference_no_plot(ch1,ch2,cutoff_area)
        chs.append(ch1)
        erosions.append(erosion)
        jumps.append(jump)
        cutoffs.append(cutoff)
    chs.append(ch2) # append last channel
    print('create list of merged channels...')
    for i in trange(len(dates)-1): # create list of merged channels
        if i == 0: 
            all_ch = chs[len(dates)-1]
        else:
            all_ch = all_ch.union(chs[len(dates)-i-1])
        all_chs.append(all_ch)
    print('create bars...')
    for i in trange(len(dates)-1): # create scroll bars
        bar = chs[i].difference(all_chs[len(dates)-i-2]) # scroll bar defined by difference
        bars.append(bar)
        color = cmap(i/float(len(dates)-1))
        for b in bar: # plotting
            if MultiPolygon(cutoffs[i]).is_valid: # sometimes this is invalid
                if not b.intersects(MultiPolygon(cutoffs[i])):
                    ax.add_patch(PolygonPatch(b,facecolor=color,edgecolor='k'))
            else:
                ax.add_patch(PolygonPatch(b,facecolor=color,edgecolor='k'))
    return bars, erosions, chs, all_chs, jumps, cutoffs

def create_channel_polygon_from_shapefile(dirname, date):
    # function for reading channel bank shapefiles and creating a polygon
    filename1 = dirname+'/lb_'+date[:4]
    filename2 = dirname+'/rb_'+date[:4]
    sf1 = shapefile.Reader(filename1).shapes()
    lb1 = np.array(sf1[0].points)
    sf2 = shapefile.Reader(filename2).shapes()
    rb1 = np.array(sf2[0].points)
    coords = []
    xm = np.hstack((lb1[:,0],rb1[::-1,0]))
    ym = np.hstack((lb1[:,1],rb1[::-1,1]))
    for i in range(len(xm)):
        coords.append((xm[i],ym[i]))
    ch = Polygon(LinearRing(coords))
    if not ch.is_valid:
        ch = ch.buffer(0)
    return ch

def create_channel_polygon(lbx,lby,rbx,rby):
    # function for creating a channel polygon
    coords = []
    xm = np.hstack((lbx,rbx[::-1]))
    ym = np.hstack((lby,rby[::-1]))
    for i in range(len(xm)):
        coords.append((xm[i],ym[i]))
    ch = Polygon(LinearRing(coords))
    return ch

def one_step_difference_no_plot(ch1, ch2, cutoff_area):
    both_channels = ch1.union(ch2) # union of the two channels
    outline = Polygon(LinearRing(list(both_channels.exterior.coords))) # outline of the union
    jump = outline.difference(both_channels) # gaps between the channels
    bar = ch1.difference(ch2) # the (point) bars are the difference between ch1 and ch2
    bar = bar.union(jump) # add gaps to bars
    erosion = ch2.difference(ch1) # erosion is the difference between ch2 and ch1
    bar_no_cutoff = list(bar.geoms) # create list of bars (cutoffs will be removed later)
    erosion_no_cutoff = list(erosion.geoms) # create list of eroded areas (cutoffs will be removed later)
    if type(jump)==MultiPolygon: # create list of gap polygons (if there is more than one gap)
        jump_no_cutoff = list(jump.geoms)
    else:
        jump_no_cutoff = jump
    cutoffs = []
    for b in bar.geoms:
        if b.area>cutoff_area:
            bar_no_cutoff.remove(b) # remove cutoff from list of bars
            for e in erosion.geoms: # remove 'fake' erosion related to cutoffs
                if b.intersects(e): # if bar intersects erosional area
                    if type(b.intersection(e))==MultiLineString:
                        erosion_no_cutoff.remove(e)
            # deal with gaps between channels:
            if type(jump)==MultiPolygon:
                for j in jump.geoms:
                    if b.intersects(j):
                        if (type(j.intersection(b))==Polygon) & (j.area>0.3*cutoff_area):
                            jump_no_cutoff.remove(j) # remove cutoff-related gap from list of gaps
                            cutoffs.append(b.symmetric_difference(b.intersection(j))) # collect cutoff
            if type(jump)==Polygon:
                if b.intersects(jump):
                    if type(jump.intersection(b))==Polygon:
                        jump_no_cutoff = []
                        cutoffs.append(b.symmetric_difference(b.intersection(jump))) # collect cutoff
    bar = MultiPolygon(bar_no_cutoff)
    erosion = MultiPolygon(erosion_no_cutoff)
    if type(jump_no_cutoff)==list:
        jump = MultiPolygon(jump_no_cutoff)
    ch1 = ch1.union(jump)
    eps = 0.1 # this is needed to get rid of 'sliver geometries' - 
    ch1 = ch1.buffer(eps, 1, join_style=2).buffer(-eps, 1, join_style=2)
    return ch1, bar, erosion, jump, cutoffs

def compute_s_coord(x,y):             
    dx = np.diff(x); dy = np.diff(y)      
    ds = np.sqrt(dx**2+dy**2)
    s = np.hstack((0,np.cumsum(ds)))
    return dx, dy, ds, s


def create_channel_segment_polygons(x, y, rbx, rby, lbx, lby, lbw, rbw, deltas, extra_width):
    x1 = x.copy()
    y1 = y.copy()
    x2 = x.copy()
    y2 = y.copy()
    dx,dy,ds,s = compute_s_coord(x,y)

    # x1,x2,y1,y2 are coordinates of points that are extra_width * deltas m beyond the channel banks on both sides:
    x1[1:-1] = x[1:-1] - (lbw[1:-1]+extra_width)*(dy[1:]+dy[:-1])/deltas # left bank
    y1[1:-1] = y[1:-1] + (lbw[1:-1]+extra_width)*(dx[1:]+dx[:-1])/deltas # left bank
    x2[1:-1] = x[1:-1] + (rbw[1:-1]+extra_width)*(dy[1:]+dy[:-1])/deltas # right bank
    y2[1:-1] = y[1:-1] - (rbw[1:-1]+extra_width)*(dx[1:]+dx[:-1])/deltas # right bank
    x1[0] = x[0] - (lbw[0]+extra_width)*dy[0]/deltas      # first point
    y1[0] = y[0] + (lbw[0]+extra_width)*dx[0]/deltas
    x2[0] = x[0] + (rbw[0]+extra_width)*dy[0]/deltas
    y2[0] = y[0] - (rbw[0]+extra_width)*dx[0]/deltas 
    x1[-1] = x[-1] - (lbw[-1]+extra_width)*dy[-1]/deltas  # last point
    y1[-1] = y[-1] + (lbw[-1]+extra_width)*dx[-1]/deltas
    x2[-1] = x[-1] + (rbw[-1]+extra_width)*dy[-1]/deltas
    y2[-1] = y[-1] - (rbw[-1]+extra_width)*dx[-1]/deltas

    polys = []
    cline = LineString(np.vstack((x,y)).T) # create linestring from centerline
    for i in trange(0,len(x1)-1):
        # create polygon:
        poly = Polygon(LinearRing([[x2[i],y2[i]],[x2[i+1],y2[i+1]],[x1[i+1],y1[i+1]],[x1[i],y1[i]]]))
        if not poly.is_valid: # if there are no self-intersections, the polygon is already 'fixed'
            fixed_polys = get_rid_of_self_intersections(poly) # otherwise remove self intersections
            fixed_polys1 = [] # 'fixed_polys' is a generator, but we need a list
            for fpoly in fixed_polys:
                fixed_polys1.append(fpoly)
            # select the polygon that intersects the centerline, get rid of the other one:
            if (fixed_polys1[0].intersects(cline)) and (not fixed_polys1[1].intersects(cline)):
                poly = fixed_polys1[0]
            if (not fixed_polys1[0].intersects(cline)) and (fixed_polys1[1].intersects(cline)):       
                poly = fixed_polys1[1]
            # if both polygons intersect the centerline:
            if (fixed_polys1[0].intersects(cline)) and (fixed_polys1[1].intersects(cline)):
                if fixed_polys1[0].intersects(prev_poly):
                    poly = fixed_polys1[0]
                else:
                    poly = fixed_polys1[1]
    #             if fixed_polys1[0].area > fixed_polys1[1].area:
    #                 poly = fixed_polys1[0]
    #             else:
    #                 poly = fixed_polys1[1]
        prev_poly = poly # store current polygon
        polys.append(poly)

    # create channel polygon:
    coords = []
    xm = np.hstack((lbx,rbx[::-1]))
    ym = np.hstack((lby,rby[::-1]))
    for i in range(len(xm)):
        coords.append((xm[i],ym[i]))
    ch = Polygon(LinearRing(coords))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    ax.plot(rbx,rby,'k')
    ax.plot(lbx,lby,'k')
    # plot all polygons:
    for poly in polys:
        ax.add_patch(PolygonPatch(poly,facecolor='none',edgecolor='k'))
    plt.axis('equal')
    return polys

def estimate_half_widths(x, y, rbx, lbx, rby, lby):
    # do the corelation (this will take a few seconds):
    pr,qr,smr = correlate_curves(x,rbx,y,rby)
    pl,ql,sml = correlate_curves(x,lbx,y,lby)
    pnr = np.delete(np.array(pr),np.where(np.diff(pr)==0)[0]+1)
    qnr = np.delete(np.array(qr),np.where(np.diff(pr)==0)[0]+1)
    pnl = np.delete(np.array(pl),np.where(np.diff(pl)==0)[0]+1)
    qnl = np.delete(np.array(ql),np.where(np.diff(pl)==0)[0]+1)
    # find left- and right-widths:
    rbw = smr[pnr,qnr]
    lbw = sml[pnl,qnl]
    # rbw and lbw are flipped relative to the centerline, so they need to be inverted:
    rbw = rbw[::-1]
    lbw = lbw[::-1]
    # plotting for QC:
    plt.figure()
    plt.plot(rbx,rby,'k')
    plt.plot(lbx,lby,'k')
    plt.plot(x,y,'r')
    for i in range(len(pnr)):
        plt.plot([x[pnr[i]], rbx[qnr[i]]], [y[pnr[i]], rby[qnr[i]]], 'b', linewidth = 0.5 )
    for i in range(len(pnl)):
        plt.plot([x[pnl[i]], lbx[qnl[i]]], [y[pnl[i]], lby[qnl[i]]], 'r', linewidth = 0.5 )
    plt.axis('equal');
    return rbw, lbw, pnr, qnr, pnl, qnl

def crop_polygons_to_channel_width(polys, ch):
    cropped_polys = [] # list for polygons that are cropped to the actual channel width
    for poly in tqdm(polys):
        cropped_polys.append(poly.intersection(ch)) # cropping

    # remove objects that are not polygons or multipolygons: 
    polys_to_be_removed = []
    # ind = 0       
    for poly in cropped_polys: 
        if (type(poly)!=Polygon) & (type(poly)!=MultiPolygon):
            polys_to_be_removed.append(poly)
        if poly.area<1.0:
            polys_to_be_removed.append(poly) 
        # ind += 1
    cropped_polys = [poly for poly in cropped_polys if poly not in polys_to_be_removed]


    # remove unnecessary small bits that are in multipolygons:
    for i in range(len(cropped_polys)):
        if type(cropped_polys[i])==MultiPolygon:
            polys_temp = list(cropped_polys[i])
            if polys_temp[0].area>=polys_temp[1].area:
                cropped_polys[i] = polys_temp[0]
            else:
                cropped_polys[i] = polys_temp[1]

    # for poly in cropped_polys:
    #     if poly.area<1.0:
    #         cropped_polys.remove(poly)
    return cropped_polys

def find_overlapping_polys(polys,crit_area):
    """function for finding overlapping polygons"""
    inds = []
    ind = 0
    pbar = tqdm(total = len(polys)/5)
    while ind<len(polys)-50: # look at 50 consecutive polygons at a time (otherwise it takes a long time)
        for ind1,ind2 in itertools.combinations(np.arange(ind,ind+50), 2):
            geom1 = polys[ind1]
            geom2 = polys[ind2]
            if geom1.intersection(geom2).area>crit_area:
                inds.append(ind1)
                inds.append(ind2)
        ind = ind+5
        pbar.update(1)
    inds = np.array(inds)
    inds = np.unique(inds)
    return inds

def repolygonize_bend(cropped_polys, cropped_polys_new, i1, i2, pad, crit_dist, new_poly_inds, x, y):
    """function for generating new, non-overlapping polygons in sharp bends
    inputs:
    cropped_polys - list of polygons that describe the channel
    cropped_polys_new - list of new polygons that do not overlap
    i1 - index of starting point of segment with overlapping polygons
    i2 - index of ending point of segment with overlapping polygons
    pad - number of polygons you want the segment to be padded with
    new_poly_inds - indices of fixed polygons
    x - 
    y - 
    outputs:
    bend - polygon that describes the fixed segment
    x1 - new x coordinates of the (fixed) left bank
    x2 - new x coordinates of the (fixed) right bank
    y1 - new y coordinates of the (fixed) left bank
    y2 - new y coordinates of the (fixed) right bank
    """
    
    # bend = cropped_polys[i1-pad] # start bend with first polygon
    # count = 1
    # for poly in cropped_polys[i1-pad+1:i2+pad]: # add all the polygons to the bend
    #     bend = bend.union(poly)
    #     count = count+1

    bend = unary_union(cropped_polys[i1-pad:i2+pad])
    count = len(cropped_polys[i1-pad:i2+pad])

    eps = 0.1 # this is needed to get rid of 'sliver geometries' 
    bend = bend.buffer(eps, 1, join_style=2).buffer(-eps, 1, join_style=2)
            
    xbend = bend.exterior.xy[0] # x coordinates of polygon that describes the bend
    ybend = bend.exterior.xy[1] # y coordinates of polygon that describes the bend
    dx, dy, ds, s = compute_s_coord(xbend,ybend) # get distances between consecutive points
    
    if len(np.where(np.abs(ds)>crit_dist)[0])==2: # if 'xbend' starts at a 'corner' point of the bend
        ind1,ind2 = np.where(np.abs(ds)>crit_dist)[0]
    else: # if 'xbend' does not start at a 'corner' point of the bend
        ind1,ind2,ind3 = np.where(np.abs(ds)>crit_dist)[0]   
    # ind1, ind2 are the indices where 'xbend' and 'ybend' switch from one bank to the other

    # coordinates of the right bank:
    b1_rbx = np.hstack((xbend[ind2+1:],xbend[1:ind1+1]))
    b1_rby = np.hstack((ybend[ind2+1:],ybend[1:ind1+1]))

    # coordinates of the left bank:
    b1_lbx = xbend[ind1+1:ind2+1]
    b1_lby = ybend[ind1+1:ind2+1]

    # resample left bank:
    tck, u = scipy.interpolate.splprep([b1_lbx,b1_lby],s=1) # parametric spline representation of curve
    unew = np.linspace(0,1,count+1) # vector for resampling
    out = scipy.interpolate.splev(unew,tck) # resampling
    b1_lbxs = out[0]
    b1_lbys = out[1]

    # resample right bank:
    tck, u = scipy.interpolate.splprep([b1_rbx,b1_rby],s=1) # parametric spline representation of curve
    unew = np.linspace(0,1,count+1) # vector for resampling
    out = scipy.interpolate.splev(unew,tck) # resampling
    b1_rbxs = out[0]
    b1_rbys = out[1]

    direction_flag = 0 # if direction of banks is same as that of flow
    x1 = b1_lbxs
    x2 = b1_rbxs[::-1]
    y1 = b1_lbys
    y2 = b1_rbys[::-1]
    
    dx1 = x1[-1]-x1[0]
    dy1 = y1[-1]-y1[0]
    dx = x[i2]-x[i1]
    dy = y[i2]-y[i1]
    
    # if direction of banks is flipped relative to flow, the coordinate arrays need to be flipped:
    if np.sign(dy1) != np.sign(dy):
        direction_flag = 1
        x1 = x1[::-1]
        x2 = x2[::-1]
        y1 = y1[::-1]
        y2 = y2[::-1]

    new_polys = [] # create new polygons for the bend
    for i in range(0,len(b1_lbxs)-1):
        if direction_flag == 0: # direction of banks is same as that of flow
            poly = Polygon(LinearRing([[x1[i+1],y1[i+1]],[x2[i+1],y2[i+1]],[x2[i],y2[i]],[x1[i],y1[i]]]))
        else: # direction of banks is flipped relative to flow
            poly = Polygon(LinearRing([[x2[i+1],y2[i+1]],[x1[i+1],y1[i+1]],[x1[i],y1[i]],[x2[i],y2[i]]]))
        new_polys.append(poly)

    # plot bend and new bend polygons:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for poly in cropped_polys[i1-pad+1:i2+pad]:
        ax.add_patch(PolygonPatch(poly,facecolor='none',edgecolor='b'))
    for poly in new_polys:
        ax.add_patch(PolygonPatch(poly,facecolor='none',edgecolor='r'))
    plt.axis('equal');
    
    plt.plot(xbend,ybend,'k.')
    plt.plot(xbend[ind1],ybend[ind1],'ro')
    plt.plot(xbend[ind2],ybend[ind2],'bo')
    if len(np.where(np.abs(ds)>100)[0])>2:
        plt.plot(xbend[ind3],ybend[ind3],'go')
    
    count = 0
    for i in np.arange(i1-pad,i2+pad):
        cropped_polys_new[i] = new_polys[i-(i1-pad)]
        new_poly_inds.append(i)
        count = count+1
    
    return bend, x1, x2, y1, y2

def simplify_polygon(poly, deltas):
    x1 = np.array(poly.exterior.xy[0])
    y1 = np.array(poly.exterior.xy[1])
    dx, dy, ds, s = compute_s_coord(x1,y1)
    ds_inds = np.where(ds>4*deltas)[0]
    corner_inds = np.sort(np.hstack((ds_inds,ds_inds+1)))
    x1 = x1[corner_inds]
    y1 = y1[corner_inds]
    dx, dy, ds, s = compute_s_coord(x1,y1)
    while len(np.where(ds==0)[0])>0: # eliminate duplicate points
        zero_ind = np.where(ds==0)[0][0]
        x1 = np.hstack((x1[:zero_ind], x1[zero_ind+1:]))
        y1 = np.hstack((y1[:zero_ind], y1[zero_ind+1:]))
        dx, dy, ds, s = compute_s_coord(x1,y1)
    poly = Polygon(LinearRing(np.vstack((x1,y1)).T))
    return poly

def simplify_all_polygons(polys, deltas):
    for i in range(len(polys)):
        poly = polys[i]
        if type(poly)==Polygon:
            if len(poly.exterior.xy[0])>5: # if polygon has more than 4 points
                polys[i] = simplify_polygon(poly, deltas=deltas)
    return polys

def create_new_bank_coordinates(cropped_polys_new, x, y):
    ds = [] # 
    for ind in range(len(cropped_polys_new)):
        poly = cropped_polys_new[ind]
        poly = orient(poly,sign=-1.0)
        x1 = np.array(poly.exterior.xy[0])[0]
        y1 = np.array(poly.exterior.xy[1])[0]
        d = (x1-x[ind])*(y[ind+1]-y[ind])-(y1-y[ind])*(x[ind+1]-x[ind])
        ds.append(d)
    # create new x and y coordinate arrays for the banks
    rbxn = []
    rbyn = []
    lbxn = []
    lbyn = []
    # start with first two points on first polygon:
    poly = cropped_polys_new[0]
    if ds[0]<0:
        rbxn.append(poly.exterior.xy[0][3])
        rbyn.append(poly.exterior.xy[1][3])
        lbxn.append(poly.exterior.xy[0][2])
        lbyn.append(poly.exterior.xy[1][2])
    else:
        rbxn.append(poly.exterior.xy[0][1])
        rbyn.append(poly.exterior.xy[1][1])
        lbxn.append(poly.exterior.xy[0][0])
        lbyn.append(poly.exterior.xy[1][0])
    # then add the rest:
    for i in range(len(cropped_polys_new)):
        poly = cropped_polys_new[i]
        if type(poly)==Polygon:
            poly = orient(poly,sign=-1.0)
            if ds[i]<0:
                rbxn.append(poly.exterior.xy[0][0])
                rbyn.append(poly.exterior.xy[1][0])
                lbxn.append(poly.exterior.xy[0][1])
                lbyn.append(poly.exterior.xy[1][1])
            else:
                rbxn.append(poly.exterior.xy[0][2])
                rbyn.append(poly.exterior.xy[1][2])
                lbxn.append(poly.exterior.xy[0][3])
                lbyn.append(poly.exterior.xy[1][3])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x,y,'b.-')
    for i in range(len(cropped_polys_new)):
        if type(cropped_polys_new[i])==Polygon:
            ax.add_patch(PolygonPatch(cropped_polys_new[i],facecolor='none',edgecolor='k'))
    plt.plot(rbxn,rbyn,'.-')
    plt.plot(lbxn,lbyn,'.-')
    plt.axis('equal');
    return rbxn, rbyn, lbxn, lbyn

def get_bti_polys(dates, dirname, ts1, ts2, deltas, W, kl):
    # fw = 'fake width' needed to create wide channel segments
    # read the centerline shapefiles for two timesteps
    date1 = dates[ts1]
    date2 = dates[ts2]
    filename1 = dirname + 'cline_'+date1[:4]
    filename2 = dirname + 'cline_'+date2[:4]
    sf1 = shapefile.Reader(filename1).shapes()
    cl1 = np.array(sf1[0].points)
    sf2 = shapefile.Reader(filename2).shapes()
    cl2 = np.array(sf2[0].points)
    
    x = cl1[:,0]
    y = cl1[:,1]
    xn = cl2[:,0]
    yn = cl2[:,1]

    curv, s = compute_curvature(x, y)
    curv = savgol_filter(curv,71,3)

    age1 = convert_string_to_date(date1)
    age2 = convert_string_to_date(date2)
    d = age2-age1
    years = d.days/365.0
    migr_rate, migr_sign, p, q = get_migr_rate(x, xn, y, yn, years)
    migr_rate = medfilt(savgol_filter(migr_rate,51,3),kernel_size=5) 

    f = 0.5
    filename = dirname + 'polys_'+date1[:4]
    sf = shapefile.Reader(filename).shapes()
    polys = []
    for i in range(0,len(sf)):
        poly = np.array(sf[i].points)
        x0 = poly[0,0]; y0 = poly[0,1]; x1 = poly[1,0]; y1 = poly[1,1];
        x2 = poly[2,0]; y2 = poly[2,1]; x3 = poly[3,0]; y3 = poly[3,1];
        xa = (1+f)*x0 - f*x1
        ya = (1+f)*y0 - f*y1
        xb = (1+f)*x1 - f*x0
        yb = (1+f)*y1 - f*y0
        xc = (1+f)*x3 - f*x2
        yc = (1+f)*y3 - f*y2
        xd = (1+f)*x2 - f*x3
        yd = (1+f)*y2 - f*y3
        poly = Polygon(LinearRing([[xa,ya],[xb,yb],[xd,yd],[xc,yc]]))
        polys.append(poly)

    bti = W*curv*migr_rate/kl
    return x, xn, y, yn, polys, bti, curv, migr_rate, s

class Bar:
    def __init__(self,age,scrolls):
        self.age = age
        self.scrolls = scrolls
    def plot(self,ax):
        for scroll in self.scrolls:
            ax.add_patch(ax.add_patch(PolygonPatch(scroll.polygon,edgecolor='k',facecolor=sns.xkcd_rgb["light gold"])))
        plt.axis('equal')
    def plot_bti(self,ax,vmin,vmax,cmap,linewidth,edgecolor):
        for scroll in self.scrolls:
            scroll.plot_bti(ax,vmin,vmax,cmap,linewidth,edgecolor)
    
class Scroll:
    def __init__(self,polygon,age,bti_polys):
        self.polygon = polygon
        self.age = age
        self.bti_polys = bti_polys
        self.area = self.polygon.area
    def plot(self,ax):
        ax.add_patch(PolygonPatch(self.polygon,edgecolor='k',facecolor=sns.xkcd_rgb["light gold"]))
    def plot_bti(self,ax,vmin,vmax,cmap,linewidth,edgecolor):
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        for i in range(len(self.bti_polys)):
            ax.add_patch(PolygonPatch(self.bti_polys[i].polygon,
                    facecolor=m.to_rgba(self.bti_polys[i].bti),edgecolor=edgecolor,linewidth=linewidth))
        
class BTI_poly:
    def __init__(self,polygon,bti):
        self.polygon = polygon
        self.bti = bti
        self.area = self.polygon.area
        
def get_rid_of_self_intersections(poly):
    ext = poly.exterior
    mls = ext.intersection(ext)
    polygons = polygonize(mls)
    return polygons

def create_bti_polys(scroll,polys,bti):
    sel_polys = [] # polygons that intersect the scroll of interest
    btis = [] # bti values that go with the polygons of interest
    for i in range(len(polys)):
        if polys[i].intersects(scroll):
            sel_polys.append(polys[i])
            btis.append(bti[i+1])
    # some polygons have self-intersections that need to be removed:
    sel_polys_fixed = []
    for poly in sel_polys:
        if poly.is_valid: # if there are no self-intersections, the polygon is already 'fixed'
            sel_polys_fixed.append(poly)
        else: # if there are self-intersections:
            fixed_polys = get_rid_of_self_intersections(poly)
            fixed_polys1 = [] # 'fixed_polys' is a generator, but we need a list
            for fpoly in fixed_polys:
                fixed_polys1.append(fpoly)
            # select the larger polygon:
            if fixed_polys1[0].area>fixed_polys1[1].area:
                sel_polys_fixed.append(fixed_polys1[0])
            else:
                sel_polys_fixed.append(fixed_polys1[1])
    sel_polys = sel_polys_fixed
    # now we are ready to clip the larger polygons in 'sel_polys' to the extent of the scroll:
    bti_polys = []
    for i in range(len(sel_polys)):
        poly = scroll.intersection(sel_polys[i])
        if type(poly)==Polygon:
            bti_polys.append(poly)
    return bti_polys, btis

def create_bar_hierarchy(bars, cutoffs, dates, all_polys, all_btis):
    scrolls = [] # list of all scroll bar polygons
    ages = []
    areas = []
    eroded_cutoffs = []
    cutoff_ages = []
    for i in range(len(bars)):
        for j in range(len(bars[i])): # bars in time step i
            # if bar does not intersect any of the cutoffs and is larger than 1 square meter:
            if (not bars[i][j].intersects(MultiPolygon(cutoffs[i]))) & (bars[i][j].area>1.0):
                scrolls.append(bars[i][j]) # append bar to list of scrolls
                ages.append(dates[i+1]) # append age of bar to list of ages
                areas.append(bars[i][j].area) # append area of bar to list of areas
            # if bar intersects any of the cutoffs and is larger than 1 square meter:
            elif (bars[i][j].intersects(MultiPolygon(cutoffs[i]))) & (bars[i][j].area>1.0):
                eroded_cutoffs.append(bars[i][j]) # append 'bar' to list of eroded cutoffs 
                cutoff_ages.append(dates[i+1]) # append age of 'bar' to list of ages of eroded cutoffs
    Bars = [] # list of bars
    for i in trange(len(bars)):
        age = dates[i+1]
        inds = np.where(np.array(ages)==age)[0]
        scroll_objects = []
        scrolls_same_age = []
        for ind in inds:
            if scrolls[ind].area>1.0:
                scrolls_same_age.append(scrolls[ind])
        for scroll in scrolls_same_age:
            bti_polys, btis = create_bti_polys(scroll, all_polys[i], all_btis[i])
            bti_poly_objects = []
            for j in range(len(bti_polys)):
                bti_poly = BTI_poly(bti_polys[j], btis[j]) # create BTI_poly object
                bti_poly_objects.append(bti_poly)
            scroll_object = Scroll(scroll, age, bti_poly_objects) # create Scroll object
            scroll_objects.append(scroll_object)
        bar = Bar(age, scroll_objects) # create Bar object
        Bars.append(bar) # add bar to the list of bars
    bti_ages = []
    bti_areas = []
    bti_polys = []
    bti_indices = []
    for bar in Bars:
        for scroll in bar.scrolls:
            for bti_poly in scroll.bti_polys:
                bti_ages.append(scroll.age)
                bti_areas.append(bti_poly.area)
                bti_polys.append(bti_poly.polygon)
                bti_indices.append(bti_poly.bti)
    return Bars, bti_ages, bti_areas, bti_polys, bti_indices

def plot_scroll_bars(bars, cutoffs, dates):
    """function for plotting scroll bars colored by age
    :param bars:
    :param cutoffs:
    :param dates:
    """
    fig = plt.figure()
    ax = plt.subplot(111)
    cmap = mpl.cm.get_cmap('viridis')
    for i in trange(len(dates)-1):
        color = cmap(i/float(len(dates)-1))
        for b in bars[i]:
            if MultiPolygon(cutoffs[i]).is_valid: # sometimes this is invalid
                if not b.intersects(MultiPolygon(cutoffs[i])):
                    ax.add_patch(PolygonPatch(b,facecolor=color,edgecolor='k'))
            else:
                ax.add_patch(PolygonPatch(b,facecolor=color,edgecolor='k'))
    plt.axis('equal');
    return fig

def plot_btis(Bars, lw = 0.1, vmin = -1, vmax = 1):
    """function for plotting bar type indices on a map

    :param Bars: name of the well (usually this is the UWI)
    :param lw: linewidth to be used when plotting BTI polygons (default is 0.1)
    :param vmin: minimum value for BTI colormap (default is -1.0)
    :param vmax: maximum value for BTI colormap (default is 1.0)"""

    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9]) # [left, bottom, width, height]
    # ax = fig.add_subplot(111)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.RdBu_r
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for Bar in tqdm(Bars):
        for scroll in Bar.scrolls:
            for bti_poly in scroll.bti_polys:
                if bti_poly.polygon.area > 1.0:
                    ax.add_patch(PolygonPatch(bti_poly.polygon, facecolor=m.to_rgba(bti_poly.bti), linewidth=lw))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical', label='bar type index')
    ax.plot(bti_poly.polygon.exterior.xy[0], bti_poly.polygon.exterior.xy[1], color = 'k', linewidth=lw)
    ax.set_aspect('equal')
    return fig