import numpy as np
# import matplotlib
# # matplotlib.rcParams.update({
# #     'font.family': 'cmss10',
# #     'axes.formatter.use_mathtext': True})
import matplotlib.pyplot as plt
from pylab import *

#%%

def set_plot_params(case, display_fonts = False):
    
    import matplotlib.font_manager as fm
    # Collect all the font names available to matplotlib
    font_names = [f.name for f in fm.fontManager.ttflist]
    if display_fonts == True:
        print(font_names)
        
    p = physical_constants()
    gr = p['golden_ratio']
    
    colors = color_dictionary()
    
    plt.rcParams['figure.max_open_warning'] = 100
    
    # plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
    #                                                     colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
    #                                                     colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
    #                                                     colors['red4'],colors['red3'],colors['red2'],colors['red1'],
    #                                                     colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
    #                                                     colors['green4'],colors['green3'],colors['green2'],colors['green1'],
    #                                                     colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
    #                                                     colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1']])
    
    # plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
    #                                                     colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
    #                                                     colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
    #                                                     colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5']])  
    
    plt.rcParams['axes.prop_cycle'] = cycler('color', colors_gist(np.linspace(.1, 1,9)))
    
    if case == 'presentation':
    
        major_tick_length = 5
        major_tick_width = 0.75
        minor_tick_width = 0.45
        tn = 4*1.1*8.6/2.54 # for figure size    
                
        plt.rcParams['figure.autolayout'] = False # True
        
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['lines.markersize'] = 5
        
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.labelpad'] = 0
        plt.rcParams['axes.xmargin'] = 0.0 # space between traces and axes
        plt.rcParams['axes.ymargin'] = 0.05
        plt.rcParams['axes.titlepad'] = 0
        
        plt.rcParams['legend.loc'] = 'best'
        
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.format'] = 'pdf'
        plt.rcParams['savefig.pad_inches'] = 0
        
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['xtick.major.bottom'] = True
        plt.rcParams['xtick.major.top'] = True
        plt.rcParams['xtick.major.size'] = major_tick_length
        plt.rcParams['xtick.major.width'] = major_tick_width
        plt.rcParams['xtick.minor.visible'] = True
        plt.rcParams['xtick.minor.size'] = major_tick_length/gr
        plt.rcParams['xtick.minor.width'] = minor_tick_width
        
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['ytick.major.left'] = True
        plt.rcParams['ytick.major.right'] = True
        plt.rcParams['ytick.major.size'] = major_tick_length
        plt.rcParams['ytick.major.width'] = major_tick_width
        plt.rcParams['ytick.minor.visible'] = True
        plt.rcParams['ytick.minor.size'] = major_tick_length/gr
        plt.rcParams['ytick.minor.width'] = minor_tick_width
        
        # plt.rcParams['font.family'] = ['sans-serif']
        # plt.rcParams['font.sans-serif'] = 'dejavuserif' # 'Verdana'#'Computer Modern Sans Serif'
        # plt.rcParams['font.serif'] = 'dejavuserif' # 'Verdana'#'Computer Modern Sans Serif'
        # plt.rcParams['font.sans-serif'] = 'cmss10' # 'Verdana'#'Computer Modern Sans Serif'
        # plt.rcParams['font.serif'] = 'cmr10' # 'Verdana'#'Computer Modern Sans Serif'
                
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['mathtext.rm'] = 'dejavuserif'
        plt.rcParams['mathtext.it'] = 'dejavuserif'
        plt.rcParams['mathtext.bf'] = 'dejavuserif'
        
        # plt.rcParams['mathtext.fontset'] = 'cm'
        # plt.rcParams['mathtext.rm'] = 'cmr10'
        # plt.rcParams['mathtext.it'] = 'cmtt10'
        # plt.rcParams['mathtext.bf'] = 'cmb10'

        # plt.rcParams['mathtext.rm'] = 'serif'
        plt.rcParams['figure.figsize'] = [tn,tn/gr]
        plt.rcParams['figure.titlesize'] = 16
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['figure.autolayout'] = False    
        
        plt.rcParams['grid.linewidth'] = major_tick_width
        plt.rcParams['grid.color'] = colors['grey7']
        
    elif case == 'publication':
                
        major_tick_length = 3
        major_tick_width = 0.25
        minor_tick_width = 0.15
        tn = 1.1*8.6/2.54 # for figure size    
                
        plt.rcParams['figure.autolayout'] = False # True
        
        plt.rcParams['lines.linewidth'] = 1.0
        plt.rcParams['lines.markersize'] = 2
        
        plt.rcParams['axes.linewidth'] = 0.75
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.labelpad'] = 0
        plt.rcParams['axes.xmargin'] = 0.0 # space between traces and axes
        plt.rcParams['axes.ymargin'] = 0.05
        plt.rcParams['axes.titlepad'] = 0
        
        plt.rcParams['legend.loc'] = 'best'
        
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.format'] = 'pdf'
        plt.rcParams['savefig.pad_inches'] = 0
        
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['xtick.major.bottom'] = True
        plt.rcParams['xtick.major.top'] = True
        plt.rcParams['xtick.major.size'] = major_tick_length
        plt.rcParams['xtick.major.width'] = major_tick_width
        plt.rcParams['xtick.minor.visible'] = True
        plt.rcParams['xtick.minor.size'] = major_tick_length/gr
        plt.rcParams['xtick.minor.width'] = minor_tick_width
        
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['ytick.major.left'] = True
        plt.rcParams['ytick.major.right'] = True
        plt.rcParams['ytick.major.size'] = major_tick_length
        plt.rcParams['ytick.major.width'] = major_tick_width
        plt.rcParams['ytick.minor.visible'] = True
        plt.rcParams['ytick.minor.size'] = major_tick_length/gr
        plt.rcParams['ytick.minor.width'] = minor_tick_width
        
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = 'cmss10' # 'Verdana'#'Computer Modern Sans Serif'
        plt.rcParams['font.serif'] = 'cmr10' # 'Verdana'#'Computer Modern Sans Serif'
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['mathtext.rm'] = 'serif'
        plt.rcParams['figure.figsize'] = [tn,tn/gr]
        plt.rcParams['figure.titlesize'] = 10
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 8
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['figure.autolayout'] = False    
        
        plt.rcParams['grid.linewidth'] = major_tick_width
        plt.rcParams['grid.color'] = colors['grey7']

    return plt.rcParams['figure.figsize']


#%%

def physical_constants():

    p = dict(h = 6.62606957e-34,#Planck's constant in kg m^2/s
         hbar = 6.62606957e-34/(2*np.pi),
         hBar = 6.62606957e-34/(2*np.pi),
         hbar__eV_fs = 10.616133416243974/(2*np.pi),
         hBar__eV_fs = 10.616133416243974/(2*np.pi),
         c = 299792458,#speed of light in meters per second
         c__um_ns = 299792.458,#speed of light in meters per second
         epsilon0 = 8.854187817e-12,#permittivity of free space in farads per meter
         mu0 = 4*np.pi*1e-7,#permeability of free space in volt seconds per amp meter
         k = 1.3806e-23,#Boltzmann's constant in joules per kelvin
         kB = 1.3806e-23,#Boltzmann's constant in joules per kelvin
         kb = 1.3806e-23,#Boltzmann's constant in joules per kelvin
         kB__eV = 8.61703e-05,#Boltzmann's constant in electron volts per kelvin
         kb__ev = 8.61703e-05,#Boltzmann's constant in electron volts per kelvin
         kB_eV = 8.61703e-05,#Boltzmann's constant in electron volts per kelvin
         kb_ev = 8.61703e-05,#Boltzmann's constant in electron volts per kelvin
         e = 1.60217657e-19,#electron charge in coulombs
         m_e = 9.10938291e-31,#mass of electron in kg
         eV = 1.60217657e-19,#joules per eV
         ev = 1.60217657e-19,#joules per eV
         Ry = 9.10938291e-31*1.60217657e-19**4/(8*8.854187817e-12**2*(6.62606957e-34/2/np.pi)**3*299792458),#13.3*eV;#Rydberg in joules
         a0 = 4*np.pi*8.854187817e-12*(6.62606957e-34/2/np.pi)**2/(9.10938291e-31*1.60217657e-19**2),#estimate of Bohr radius
         Phi0 = 6.62606957e-34/(2*1.60217657e-19),#flux quantum
         Phi0__pH_ns = 6.62606957e3/(2*1.60217657),
         N_mole = 6.02214076e23, # atoms per mole
         golden_ratio = (1+np.sqrt(5))/2,
         gamma_euler = 0.5772  # Euler's constant
         )

    return p 

def material_parameters():
    
    p = physical_constants()
    
    mp = dict(epsilon_sio2 = (1.46**2)*p['epsilon0'], # dc permittivity of silicon dioxide
              epsilon_si = (3.48**2)*p['epsilon0'], # dc permittivity of silicon
              epsilon_gaas = 12.85*p['epsilon0'], # dc permittivity of GaAs
              epsilon_algaas = 12.9*p['epsilon0'], # dc permittivity of AlGaAs
              n_i__si = 1e16, # electrons per meter cubed in silicon at 300K, grimoire table 5
              n_i__gaas = 2.25e12, # electrons per meter cubed in GaAs at 300K, grimoire table 6
              Lsq__MoSi = 160e-12, # kinetic inductance per square for MoSi
              Lsq__WSi = 400e-12, # kinetic inductance per square for WSi
              rsq__MoSi = 500, # resistance per square for MoSi,
              alpha__MoSi = 2e-2, # w_wire = alpha Idi_sat
              Eg__gaas = 2.275e-19 # GaAs band gap in joules
              )
    
    return mp

#%%

def color_dictionary():

    colors = dict()    

    ## define colors
    #blues  lightest to darkest
    blueVec1 = np.array([145,184,219]); colors['blue1'] = blueVec1/256;
    blueVec2 = np.array([96,161,219]); colors['blue2'] = blueVec2/256;
    blueVec3 = np.array([24,90,149]); colors['blue3'] = blueVec3/256;
    blueVec4 = np.array([44,73,100]); colors['blue4'] = blueVec4/256;
    blueVec5 = np.array([4,44,80]); colors['blue5'] = blueVec5/256;
    colors['blue1.5'] = (blueVec1+blueVec2)/(512);
    colors['blue2.5'] = (blueVec2+blueVec3)/(512);
    colors['blue3.5'] = (blueVec3+blueVec4)/(512);
    colors['blue4.5'] = (blueVec4+blueVec5)/(512);
    #reds  lightest to darkest
    redVec1 = np.array([246,177,156]); colors['red1'] = redVec1/256;
    redVec2 = np.array([246,131,98]); colors['red2'] = redVec2/256;
    redVec3 = np.array([230,69,23]); colors['red3'] = redVec3/256;
    redVec4 = np.array([154,82,61]); colors['red4'] = redVec4/256;
    redVec5 = np.array([123,31,4]); colors['red5'] = redVec5/256;
    colors['red1.5'] = (redVec1+redVec2)/(512);
    colors['red2.5'] = (redVec2+redVec3)/(512);
    colors['red3.5'] = (redVec3+redVec4)/(512);
    colors['red4.5'] = (redVec4+redVec5)/(512);
    #greens  lightest to darkest
    greenVec1 = np.array([142,223,180]); colors['green1'] = greenVec1/256;
    greenVec2 = np.array([89,223,151]); colors['green2'] = greenVec2/256;
    greenVec3 = np.array([16,162,84]); colors['green3'] = greenVec3/256;
    greenVec4 = np.array([43,109,74]); colors['green4'] = greenVec4/256;
    greenVec5 = np.array([3,87,42]); colors['green5'] = greenVec5/256;
    colors['green1.5'] = (greenVec1+greenVec2)/(512);
    colors['green2.5'] = (greenVec2+greenVec3)/(512);
    colors['green3.5'] = (greenVec3+greenVec4)/(512);
    colors['green4.5'] = (greenVec4+greenVec5)/(512);
    #yellows  lightest to darkest
    yellowVec1 = np.array([246,204,156]); colors['yellow1'] = yellowVec1/256;
    yellowVec2 = np.array([246,185,98]); colors['yellow2'] = yellowVec2/256;
    yellowVec3 = np.array([230,144,23]); colors['yellow3'] = yellowVec3/256;
    yellowVec4 = np.array([154,115,61]); colors['yellow4'] = yellowVec4/256;
    yellowVec5 = np.array([123,74,4]); colors['yellow5'] = yellowVec5/256;
    colors['yellow1.5'] = (yellowVec1+yellowVec2)/(512);
    colors['yellow2.5'] = (yellowVec2+yellowVec3)/(512);
    colors['yellow3.5'] = (yellowVec3+yellowVec4)/(512);
    colors['yellow4.5'] = (yellowVec4+yellowVec5)/(512);
    
    #blue grays
    gBlueVec1 = np.array([197,199,202]); colors['bluegrey1'] = gBlueVec1/256;
    gBlueVec2 = np.array([195,198,202]); colors['bluegrey2'] = gBlueVec2/256;
    gBlueVec3 = np.array([142,145,149]); colors['bluegrey3'] = gBlueVec3/256;
    gBlueVec4 = np.array([108,110,111]); colors['bluegrey4'] = gBlueVec4/256;
    gBlueVec5 = np.array([46,73,97]); colors['bluegrey5'] = gBlueVec5/256;
    colors['bluegrey1.5'] = (gBlueVec1+gBlueVec2)/(512);
    colors['bluegrey2.5'] = (gBlueVec2+gBlueVec3)/(512);
    colors['bluegrey3.5'] = (gBlueVec3+gBlueVec4)/(512);
    colors['bluegrey4.5'] = (gBlueVec4+gBlueVec5)/(512);
    #red grays
    gRedVec1 = np.array([242,237,236]); colors['redgrey1'] = gRedVec1/256;
    gRedVec2 = np.array([242,235,233]); colors['redgrey2'] = gRedVec2/256;
    gRedVec3 = np.array([230,231,218]); colors['redgrey3'] = gRedVec3/256;
    gRedVec4 = np.array([172,167,166]); colors['redgrey4'] = gRedVec4/256;
    gRedVec5 = np.array([149,88,71]); colors['redgrey5'] = gRedVec5/256;
    colors['redgrey1.5'] = (gRedVec1+gRedVec2)/(512);
    colors['redgrey2.5'] = (gRedVec2+gRedVec3)/(512);
    colors['redgrey3.5'] = (gRedVec3+gRedVec4)/(512);
    colors['redgrey4.5'] = (gRedVec4+gRedVec5)/(512);
    #green grays
    gGreenVec1 = np.array([203,209,206]); colors['greengrey1'] = gGreenVec1/256;
    gGreenVec2 = np.array([201,209,204]); colors['greengrey2'] = gGreenVec2/256;
    gGreenVec3 = np.array([154,162,158]); colors['greengrey3'] = gGreenVec3/256;
    gGreenVec4 = np.array([117,122,119]); colors['greengrey4'] = gGreenVec4/256;
    gGreenVec5 = np.array([50,105,76]); colors['greengrey5'] = gGreenVec5/256;
    colors['greengrey1.5'] = (gGreenVec1+gGreenVec2)/(512);
    colors['greengrey2.5'] = (gGreenVec2+gGreenVec3)/(512);
    colors['greengrey3.5'] = (gGreenVec3+gGreenVec4)/(512);
    colors['greengrey4.5'] = (gGreenVec4+gGreenVec5)/(512);
    #yellow grays
    gYellowVec1 = np.array([242,240,236]); colors['yellowgrey1'] = gYellowVec1/256;
    gYellowVec2 = np.array([242,239,233]); colors['yellowgrey2'] = gYellowVec2/256;
    gYellowVec3 = np.array([230,225,218]); colors['yellowgrey3'] = gYellowVec3/256;
    gYellowVec4 = np.array([172,169,166]); colors['yellowgrey4'] = gYellowVec4/256;
    gYellowVec5 =np.array( [149,117,71]); colors['yellowgrey5'] = gYellowVec5/256;
    colors['yellowgrey1.5'] = (gYellowVec1+gYellowVec2)/(512);
    colors['yellowgrey2.5'] = (gYellowVec2+gYellowVec3)/(512);
    colors['yellowgrey3.5'] = (gYellowVec3+gYellowVec4)/(512);
    colors['yellowgrey4.5'] = (gYellowVec4+gYellowVec5)/(512);
    
    #pure grays (white to black)
    gVec1 = np.array([256,256,256]); colors['grey1'] = gVec1/256;
    colors['white'] = colors['grey1']
    gVec2 = np.array([242,242,242]); colors['grey2'] = gVec2/256;
    gVec3 = np.array([230,230,230]); colors['grey3'] = gVec3/256;
    gVec4 = np.array([204,204,204]); colors['grey4'] = gVec4/256;
    gVec5 = np.array([179,179,179]); colors['grey5'] = gVec5/256;
    gVec6 = np.array([153,153,153]); colors['grey6'] = gVec6/256;
    gVec7 = np.array([128,128,128]); colors['grey7'] = gVec7/256;
    gVec8 = np.array([102,102,102]); colors['grey8'] = gVec8/256;
    gVec9 = np.array([77,77,77]); colors['grey9'] = gVec9/256;
    gVec10 = np.array([51,51,51]); colors['grey10'] = gVec10/256;
    gVec11 = np.array([26,26,26]); colors['grey11'] = gVec11/256;
    gVec12 = np.array([0,0,0]); colors['grey12'] = gVec12/256;
    colors['black'] = np.array([0,0,0]);
    
    # alt blue, green, red, yellow
    alt_blue_light = np.array([162,188,200]); colors['alt_blue_light'] = alt_blue_light/256;
    alt_blue_dark = np.array([134,154,175]); colors['alt_blue_dark'] = alt_blue_dark/256;
    alt_green_light = np.array([163,185,169]); colors['alt_green_light'] = alt_green_light/256;
    alt_green_dark = np.array([117,135,133]); colors['alt_green_dark'] = alt_green_dark/256;
    alt_red_light = np.array([175,165,175]); colors['alt_red_light'] = alt_red_light/256;
    alt_red_dark = np.array([145,119,123]); colors['alt_red_dark'] = alt_red_dark/256;
    alt_yellow_light = np.array([175,165,175]); colors['alt_yellow_light'] = alt_yellow_light/256;
    alt_yellow_dark = np.array([145,119,123]); colors['alt_yellow_dark'] = alt_yellow_dark/256;
    
    return colors


    # gist_earth
def colors_gist(x): # x can be scalar or vector, index of color between 0 (white) and 1 (black)
    
    return plt.cm.gist_earth_r(x)

def index_finder(var_1,var_2):
    
    if type(var_1).__name__ == 'float' or type(var_1).__name__ == 'float64' or type(var_1).__name__ == 'int' or type(var_1).__name__ == 'uint8':
        value = var_1
        array = np.asarray(var_2)
    elif type(var_2).__name__ == 'float' or type(var_2).__name__ == 'float64' or type(var_2).__name__ == 'int' or type(var_2).__name__ == 'uint8':
        value = var_2
        array = np.asarray(var_1)
    else:
        raise ValueError('index_finder: it doesn\'t seem like either input is an integer or float. type(var_1) = {}, type(var_2) = {}'.format(type(var_1).__name__,type(var_2).__name__))
        
    if len(np.shape(array)) == 2:
        _idx_1d = ( np.abs( array[:] - value ) ).argmin()
        _inds = np.unravel_index(_idx_1d,np.shape(array))
    elif len(np.shape(array)) == 1:
        _inds = ( np.abs( array[:] - value ) ).argmin()
    
    return _inds

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(np.round(255*rgb[0])),int(np.round(255*rgb[1])),int(np.round(255*rgb[2])))
