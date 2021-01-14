# channelmapper

'channelmapper' is a Python toolset for mapping and analyzing meandering channels in satellite imagery. It accompanies the following paper: "Autogenic translation and counter point bar deposition in meandering rivers", by Z. Sylvester, P.R. Durkin, S.M. Hubbard, and D. Mohrig, in press, GSA Bulletin, v.133, doi:10.1130/B35829.1. An older version of the paper can be found [here](https://eartharxiv.org/repository/view/1003/).

There are two key uses of 'channelmapper': (1) map channels in satellite imagery, by generating pixel-precise x-y coordinates of their banks; and (2) create maps of the bars that are left behind when time-lapse data is available. The two Jupyter notebooks ('Channelmapper.ipynb' and 'Create_bars_and_BTI_polygons.ipynb') reflect these two applications. Results are saved as shapefiles and can be loaded into GIS software.

The output from mapping a channel consists of the two banks, a centerline, and a set of polygons of predefined length that describe the channel and can be used to display properties along the channel. The banks and the centerline have the same number of points and the polygons do not have self-intersections:

<p align="center">
<img src="https://github.com/zsylvester/channelmapper/blob/main/banks_and_polygons.png" width="500">
</p>

This way, high-quality measurements can be derived even in bends with high curvatures and unusual geometries like the one shown above. The ['Channelmapper.ipynb' notebook](https://github.com/zsylvester/channelmapper/blob/main/Channelmapper.ipynb) goes through the workflow of mapping a segment of the Mamore River in Bolivia, using a Landsat scene from 2018. The initial estimates of the centerline and banks are obtained using the ['rivamap'](https://github.com/isikdogan/rivamap) package; however, the rest of the workflow (for getting the polygons and the even sampling along the lines) can be applied to any kind of channel bank data.

If the migration of the same channel segment is mapped over multiple time steps, the workflow illustrated in the [second notebook](https://github.com/zsylvester/channelmapper/blob/main/Create_bars_and_BTI_polygons.ipynb) can be used to generate shapefiles that describe the deposits and cutoffs left behind. The terminology used is shown below.

<p align="center">
<img src="https://github.com/zsylvester/channelmapper/blob/main/bar_terminology.png" width="500">
</p>

In this case, the 'bar type index' parameter is used to color the bars, but other parameters such as curvature, migration rate, or age can be used as well. See the paper for more detail. 

