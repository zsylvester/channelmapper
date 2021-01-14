'channelmapper' is a Python toolset for mapping and analyzing meandering channels in satellite imagery. It accompanies the following paper: "Autogenic translation and counter point bar deposition in meandering rivers", by Z. Sylvester, P.R. Durkin, S.M. Hubbard, and D. Mohrig, in press, GSA Bulletin, v.133, doi:10.1130/B35829.1. An older version of the paper can be found [here](https://eartharxiv.org/repository/view/1003/).

There two key uses of 'channelmapper': (1) map channels in satellite imagery, by generating pixel-precise x-y coordinates of their banks; and (2) create maps of the bars that are left behind when time-lapse data is available. The two Jupyter notebooks ('Channelmapper.ipynb' and 'Create_bars_and_BTI_polygons.ipynb') reflect these two applications. Results are saved as shapefiles and can be loaded into GIS software.

The output from mapping a channel consists of the two banks, a centerline, and a set of polygons of predefined length that describe the channel and can be used to display properties along the channel. The banks and the centerline have the same number of points and the polygons do not have self-intersections:

<img src="https://github.com/zsylvester/channelmapper/blob/main/banks_and_polygons.png" width="500">

