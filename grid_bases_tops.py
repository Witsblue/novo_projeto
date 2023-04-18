import numpy as np
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.cm as cm
import importlib
import os
import pandas as pd

"""
TAREFAS:

"""

plt.ion()
plt.interactive(True)

# picking up data from files
data = pd.read_csv('data/layers.csv')

lyr_lats = np.asarray(data['latitude'])
lyr_lons = np.asarray(data['longitude'])
lyr_years = np.asarray(data['year'])
lyr_months = np.asarray(data['month'])
lyr_month_n = np.asarray(data['month_number'])
lyr_dn = np.asarray(data['day_night'])
lyr_profs = np.asarray(data['profile'])
lyr_cad = np.asarray(data['CAD'])
lyr_bases = np.asarray(data['base_altitude'])
lyr_tops = np.asarray(data['top_altitude'])
lyr_tempt = np.asarray(data['top_temperature'])
lyr_mle = np.asarray(data['min_laser_energy'])
lyr_i = np.asarray(data['i'])
lyr_j = np.asarray(data['j'])

"""
Número de células em latitude: 10 (1.5 grau cada)
Número de células em longitude: 8 (3 graus cada)
Número de instantes de tempo: 6 (meses)
"""

iyear = 2009
fyear = 2016
monthlist_tot = np.arange(1,13,1)  # array with all months (1, 2, 3,...)
monthlist_n = np.arange(37,121,1)
monthlist_wet = [1,2,3,4] # list with months from wet season
monthlist_dry = [6,7,8,9] # list with months from dry season

# sets grid cells and creates the grid
ilat, ilon = 2.5, -74.
flat, flon = -12.5, -50.
itime = iyear + 0.5/12.  # first time instant considered in the base grid
dtime = 1./12.  # increment of time for the base grid
nlatcell = 10   # number of latitude increments
nloncell = 8    # number of longitude increments

dlat = (flat - ilat) / nlatcell # dlat = 1.5
dlon = (flon - ilon) / nloncell # dlon = 3

flag_dn = ''

u = 0 # counting variable, just to know how much cells were computed

# GENERATING THE BASE GRID ----------

profs = [ [ [] for j in range(nloncell)] for i in range(nlatcell) ] # the base grid itself

# looping over each cell

for i in range(nlatcell):
    
    for j in range(nloncell):
    
        # lists of variables to pick up the yearly means
        bases_year_wet = []
        bases_year_dry = []
        tops_year_wet = []
        tops_year_dry = []
    
        for year in range(iyear, fyear+1):
            
            if year==2009:
            
                # mask for cirrus layers inside that call in dry season
                mask_dry = (lyr_years==year) & (lyr_months>=6) & (lyr_months<=9) & (lyr_mle>0.08) & (lyr_i==i) & (lyr_j==j) & (lyr_cad>=70) & (lyr_cad<=100) & (lyr_tempt<=-37) & (lyr_bases>=8)
                        
                bases_dry = lyr_bases[mask_dry==True]
                tops_dry = lyr_tops[mask_dry==True]
                     
                # takes the yearly mean and saves it in x_year_wet or x_year_dry       
                bases_year_dry.append( list(bases_dry) )
                tops_year_dry.append( list(tops_dry) )
                
            elif year==2016:
            
                # mask for cirrus layers inside that cell in wet season
                mask_wet = (lyr_years==year) & (lyr_months>=1) & (lyr_months<=4) & (lyr_mle>0.08) & (lyr_i==i) & (lyr_j==j) & (lyr_cad>=70) & (lyr_cad<=100) & (lyr_tempt<=-37) & (lyr_bases>=8)
                        
                # list of bases and tops in a specific cell and season
                bases_wet = lyr_bases[mask_wet==True]
                tops_wet = lyr_tops[mask_wet==True]
                     
                # takes the yearly mean and saves it in x_year_wet or x_year_dry       
                bases_year_wet.append( list(bases_wet) )
                tops_year_wet.append( list(tops_wet) )
                
            else:

                # mask for cirrus layers inside that cell in wet season
                mask_wet = (lyr_years==year) & (lyr_months>=1) & (lyr_months<=4) & (lyr_mle>0.08) & (lyr_i==i) & (lyr_j==j) & (lyr_cad>=70) & (lyr_cad<=100) & (lyr_tempt<=-37) & (lyr_bases>=8)
                        
                # list of bases and tops in a specific cell and season
                bases_wet = lyr_bases[mask_wet==True]
                tops_wet = lyr_tops[mask_wet==True]
                        
                # mask for cirrus layers inside that call in dry season
                mask_dry = (lyr_years==year) & (lyr_months>=6) & (lyr_months<=9) & (lyr_mle>0.08) & (lyr_i==i) & (lyr_j==j) & (lyr_cad<=100) & (lyr_tempt<=-37) & (lyr_bases>=8)
                        
                bases_dry = lyr_bases[mask_dry==True]
                tops_dry = lyr_tops[mask_dry==True]
                     
                # takes the yearly mean and saves it in x_year_wet or x_year_dry       
                bases_year_wet.append( list(bases_wet) )
                tops_year_wet.append( list(tops_wet) )
                bases_year_dry.append( list(bases_dry) )
                tops_year_dry.append( list(tops_dry) )
        
        bases_year_wet = sum(bases_year_wet, [])
        tops_year_wet = sum(tops_year_wet, [])
        bases_year_dry = sum(bases_year_dry, [])
        tops_year_dry = sum(tops_year_dry, [])
        
        # saves the means in profs (profs will have four informations for each cell)
        profs[i][j].append( np.nanmedian(bases_year_wet) )
        profs[i][j].append( np.nanmedian(bases_year_dry) )
        profs[i][j].append( np.nanmedian(tops_year_wet) )
        profs[i][j].append( np.nanmedian(tops_year_dry) )

# CREATING MAP ----------

map_bases_wet = np.zeros([nlatcell, nloncell])
map_bases_dry = np.zeros([nlatcell, nloncell])
map_tops_wet = np.zeros([nlatcell, nloncell])
map_tops_dry = np.zeros([nlatcell, nloncell])

for i in range(nlatcell):

    for j in range(nloncell):

        map_bases_wet[i][j] = profs[i][j][0]    # grid to be plotted
        map_bases_dry[i][j] = profs[i][j][1]    # grid to be plotted
        map_tops_wet[i][j] = profs[i][j][2]     # grid to be plotted
        map_tops_dry[i][j] = profs[i][j][3]     # grid to be plotted

# PLOTTING ----------

figs, axs = plt.subplots(2,2, figsize=(10,7), subplot_kw={"projection": ccrs.PlateCarree()})
plt.subplots_adjust(left=0.03, right=0.9, bottom=0.03, top=0.97, wspace=0.03, hspace=0.1)

# BASE ALTITUDES (WET)

# creating geographic map for the plot

axs[0,0].add_feature(cfeature.COASTLINE, linestyle='-')
axs[0,0].add_feature(cfeature.BORDERS, linestyle='-')
axs[0,0].add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', edgecolor='k', facecolor='none'), linestyle=':')
extent = [ilon, flon, flat, ilat]
axs[0,0].set_extent(extent)

im0 = axs[0,0].imshow(map_bases_wet, cmap='Blues', interpolation=None, extent=extent, vmin=11.5, vmax=14.4) # plotting the grid

axs[0,0].set_title('Base altitudes (wet season)')

# BASE ALTITUDES (DRY)

axs[0,1].add_feature(cfeature.COASTLINE, linestyle='-')
axs[0,1].add_feature(cfeature.BORDERS, linestyle='-')
axs[0,1].add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', edgecolor='k', facecolor='none'), linestyle=':')
axs[0,1].set_extent(extent)

im1 = axs[0,1].imshow(map_bases_dry, cmap='Blues', interpolation=None, extent=extent, vmin=11.5, vmax=14.5)

axs[0,1].set_title('Base altitudes (dry season)')

# applying the colorbar
cax1 = figs.add_axes([axs[0,1].get_position().x1+0.01, axs[0,1].get_position().y0,0.02, axs[0,1].get_position().height]) # adjust box for the colormap
#fig.colorbar(im, cax=cax, extend='max')
cbar1 = figs.colorbar(im1, cax=cax1)
cbar1.ax.set_ylabel('km', rotation=0, labelpad=10)

gl0 = axs[0,0].gridlines(draw_labels=True, linestyle='--', color='k', crs=ccrs.PlateCarree()) # add gridlines (lat-lon) to the first plot
gl0.top_labels = False  # dont want longitude numbers on the top of the plot
gl0.right_labels = False # dont want latitude numbers on the right of the plot

gl1 = axs[0,1].gridlines(draw_labels=True, linestyle='--', color='k', crs=ccrs.PlateCarree()) # add gridlines (lat-lon) to the second plot
gl1.top_labels = False
gl1.right_labels = False
gl1.left_labels = False # dont want latitude numbers on the right of the plot

# TOP ALTITUDES (WET)

axs[1,0].add_feature(cfeature.COASTLINE, linestyle='-')
axs[1,0].add_feature(cfeature.BORDERS, linestyle='-')
axs[1,0].add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', edgecolor='k', facecolor='none'), linestyle=':')
axs[1,0].set_extent(extent)

im2 = axs[1,0].imshow(map_tops_wet, cmap='Oranges', interpolation=None, extent=extent, vmin=13.3, vmax=16.5)

axs[1,0].set_title('Top altitudes (wet season)')

# TOP ALTITUDES (DRY)

axs[1,1].add_feature(cfeature.COASTLINE, linestyle='-')
axs[1,1].add_feature(cfeature.BORDERS, linestyle='-')
axs[1,1].add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', edgecolor='k', facecolor='none'), linestyle=':')
axs[1,1].set_extent(extent)

im3 = axs[1,1].imshow(map_tops_dry, cmap='Oranges', interpolation=None, extent=extent, vmin=13.3, vmax=16.5)

axs[1,1].set_title('Top altitudes (dry season)')

# applying the colorbar
cax2 = figs.add_axes([axs[1,1].get_position().x1+0.01, axs[1,1].get_position().y0,0.02, axs[1,1].get_position().height])
#fig.colorbar(im, cax=cax, extend='max')
cbar2 = figs.colorbar(im3, cax=cax2)
cbar2.ax.set_ylabel('km', rotation=0, labelpad=10)

gl2 = axs[1,0].gridlines(draw_labels=True, linestyle='--', color='k', crs=ccrs.PlateCarree()) # add gridlines (lat-lon) to the third plot
gl2.top_labels = False
gl2.right_labels = False

gl3 = axs[1,1].gridlines(draw_labels=True, linestyle='--', color='k', crs=ccrs.PlateCarree()) # add gridlines (lat-lon) to the fourth plot
gl3.top_labels = False
gl3.right_labels = False
gl3.left_labels = False

figs.savefig('grid_bases_tops_'+str(iyear)+'_'+str(fyear)+'.png')

plt.show()

