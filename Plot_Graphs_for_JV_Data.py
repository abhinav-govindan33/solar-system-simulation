"""
Program that takes initial position and velocity vectors of different celestial objects from HORIZONS Web Interface and generates position values based on the RK4 algorithm.
Compares these generated position values with those from the HORIZONS interface and gives the percentage error between the two values. 
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from scipy import integrate 
import astropy.units as u 
import astropy.constants as c 
import time 

class body() : 
    """
        Class that assigns the initial position and velocity vectors as well as the mass and name to a given celestial body
    """
    def __init__(self, x_vec, v_vec, mass, name) : 
        self.x_vec = x_vec
        self.v_vec = v_vec
        self.mass = mass
        self.name = name
        
    def return_vec(self) :
        """ Function that returns one vector containing positions and velocities  """
        y_vec = np.concatenate((self.x_vec,self.v_vec))
        return y_vec 
    
    def return_mass(self) : 
        return self.mass
    
    def return_name(self) : 
        return self.name
    
    
class sim() : 
    """
        Class that performs the simulation, with the input as "body" class 
    """
    def __init__(self, bodies) : 
        self.bodies = bodies 
        self.n_bodies = len(bodies)
        self.n_dim = 6.0 
        self.quant_vec = np.concatenate(np.array([i.return_vec() for i in self.bodies]))
        self.mass_vec = np.array([i.return_mass() for i in self.bodies])
        self.name_vec = np.array([i.return_name() for i in self.bodies])
        
    def set_eqn(self, calc_diff_eqn) : 
        """ Takes in an external function and sets that as the solver in this class """
        self.calc_diff_eqn = calc_diff_eqn
        
    def rk4(self, t, h,G) :
        """ Applies the RK4 algorithm using the equation defined previously."""
        k1 = h*self.calc_diff_eqn(t, self.quant_vec,G,self.mass_vec)
        k2 = h*self.calc_diff_eqn(t + 0.5*h , self.quant_vec + 0.5*k1 ,G, self.mass_vec)
        k3 = h*self.calc_diff_eqn(t + 0.5*h , self.quant_vec + 0.5*k2 ,G, self.mass_vec)
        k4 = h*self.calc_diff_eqn(t + h , self.quant_vec + k3 ,G, self.mass_vec)
        y_new = self.quant_vec + ((k1 + 2*k2 + 2*k3 + k4)/6)
        return y_new 
        
   
    
    def runsim(self, T, h, G,t0 = 0) : 
        """ Runs the simulation for the given amount of time and at each time point, it calls the rk4 function and updates the quant_vec with the newly generated values""" 
        self.nsteps = int((T-t0)/(h)) 
        self.hist = []
        self.hist.append(self.quant_vec)
        for i in range(self.nsteps) : 
            y_new = self.rk4(0,h,G)
            self.hist.append(y_new)
            self.quant_vec = y_new 
        self.hist = np.array(self.hist)
    
    def plot_orbit(self) : 
        """ Picks out the x, y and z values from the quant_vec and generates three separate one dimensional arrays. """
        self.x_vals = np.zeros((self.n_bodies,len(self.hist)))
        self.y_vals = np.zeros((self.n_bodies,len(self.hist)))
        self.z_vals = np.zeros((self.n_bodies,len(self.hist))) 
        self.r_vals = np.zeros((self.n_bodies,len(self.hist)))
        
        for i in range(self.n_bodies) : 
            ioff = i*6
            for j in range(len(self.hist)) : 
                self.x_vals[i][j] = self.hist[j][ioff]
                self.y_vals[i][j] = self.hist[j][ioff+1]
                self.z_vals[i][j] = self.hist[j][ioff+2]
                self.r_vals[i][j] = np.sqrt((self.x_vals[i][j]**2) + (self.y_vals[i][j]**2) + (self.z_vals[i][j]**2))
        
         
       
        
def nbody_solve(t,y, G,masses):
    """ Function that will be used to solve the gravitational differential equation """
    N_bodies = int(len(y) / 6)
    solved_vector = np.zeros(y.size)
    for i in range(N_bodies):
        ioffset = i * 6 
        for j in range(N_bodies):
            joffset = j*6
            solved_vector[ioffset] = y[ioffset+3]
            solved_vector[ioffset+1] = y[ioffset+4]
            solved_vector[ioffset+2] = y[ioffset+5]
            if i != j:
                dx = y[ioffset] - y[joffset]
                dy = y[ioffset+1] - y[joffset+1]
                dz = y[ioffset+2] - y[joffset+2] 
                r = (dx**2+dy**2+dz**2)**0.5
                ax = (-G*masses[j] / r**3) * dx
                ay = (-G*masses[j] / r**3) * dy
                az = (-G*masses[j] / r**3) * dz
                #ax = ax.value
                #ay = ay.value
                #az = az.value
                solved_vector[ioffset+3] += ax
                solved_vector[ioffset+4] += ay
                solved_vector[ioffset+5] += az            
    return solved_vector 

""" Useful Constants and Definitions """ 
G_vals = np.array([6.67430E-11])
NG = 6.6740005E-11

day = 1/(3652422/10000)
d_t = 1*day
Del_t = 10*day
t_start = 0
t_end = 220

NMSun = 1988500E24
NA = (1496/1000)*(1E11)
NY = (24*3600)/day
G_used = (G_vals*(NY**2))/(NA**3)
km = (1E3)/(NA)
kmps = km*NY
        

 """ Initial Data for the given celestial bodies """ 
#sun = body(x_vec = list(np.array([1019855.1267144, 687730.6519480,40129.1888413])*1000), v_vec = list(np.array([-0.009339943726893, 0.010936844481851, -0.001442116837987])*1000), mass = 1988500E24, name = 'Sun' )
sun = body(x_vec = list(np.array([0.,0.,0.])*km), v_vec = list(np.array([0., 0., 0.])*kmps), mass = 1988409.870967E24, name = 'Sun' )

""" Mercury Data """ 
r0_merc = [4.922691107409813E+06,  4.623377506463538E+07,  2.079199440671939E+06]
v0_merc = [-5.752757817439645E+01,  9.882556854717516E+00, -2.033365671769818E+00]
mercury = body(x_vec = list(np.array(r0_merc)*km), v_vec = list(np.array(v0_merc)*kmps), mass = 0.3301020332E24, name = 'Mercury' )


""" Venus Data """ 
r0_v = [4.117362872324210E+07, -1.007014723374994E+08,  5.818275371987320E+06]
v0_v = [3.214637737164854E+01,  1.320439651234144E+01,  1.424293962660558E+00]
venus = body(x_vec = list(np.array(r0_v)*km), v_vec = list(np.array(v0_v)*kmps), mass = 4.867311928E24, name = 'Venus' )


""" Earth data """ 
r0 = [-1.407424742173435E+08,  4.495651419958439E+07, -1.844519862557191E+07]
v0 = [-9.643050088711835E+00, -2.832107300746329E+01,  6.523114017645071E-01] 
earth = body(x_vec = list(np.array(r0)*km), v_vec = list(np.array(v0)*kmps), mass = 6.045645225E24, name = 'Earth' )


""" Mars data """ 
r0_m = [-1.965079196531301E+08, -1.240344437254834E+08, -1.380155858781657E+07]
v0_m = [1.423536316245344E+01, -1.874875220753668E+01,  1.963126602524141E+00]
mars = body(x_vec = list(np.array(r0_m)*km), v_vec = list(np.array(v0_m)*kmps), mass = 0.6416908140E24, name = 'Mars' )


""" Jupiter Data """ 
r0_jupiter = [-6.498189156151918E+08, -4.784213427978374E+08, -2.892667058338800E+07]
v0_jupiter = [7.696490519130252E+00, -9.875334294603725E+00,  1.268343426164598E+00]
jupiter    = body(x_vec = list(np.array(r0_jupiter)*km), v_vec = list(np.array(v0_jupiter)*kmps), mass = 1898.517766E24, name = 'Jupiter' )

""" Saturn Data """ 
r0_saturn = [-7.008122570693592E+08, -1.330802936471254E+09,  4.348524054186463E+07]
v0_saturn = [8.005025985703082E+00, -4.363737237588655E+00,  8.459183734892382E-01]
saturn    = body(x_vec = list(np.array(r0_saturn)*km), v_vec = list(np.array(v0_saturn)*kmps), mass = 568.4576229E24, name = 'Saturn' )


""" Uranus Data """ 
r0_uranus = [6.977695737277987E+08,  2.751163072704401E+09, -8.028746987244916E+07]
v0_uranus = [-6.690936669055445E+00,  1.344745637744120E+00, -7.394938604945339E-01]
uranus    = body(x_vec = list(np.array(r0_uranus)*km), v_vec = list(np.array(v0_uranus)*kmps), mass = 86.81858324E24, name = 'Uranus' )


""" Neptune Data """ 
r0_neptune = [-4.474961425125463E+09,  5.562144983605943E+08, -4.162897311891971E+08]
v0_neptune = [-7.163700439805839E-01, -5.350618244928026E+00,  3.444244739149327E-01]
neptune    = body(x_vec = list(np.array(r0_neptune)*km), v_vec = list(np.array(v0_neptune)*kmps), mass = 102.45214E24, name = 'Neptune' )

""" Pluto Data """ 
r0_pluto = [3.324092012988805E+09, -4.245211670416664E+09, -5.065557470396831E+08]
v0_pluto = [4.450231357145345E+00,  2.116705184374955E+00, -1.513791928229923E+00]
pluto = body(x_vec = list(np.array(r0_pluto)*1000), v_vec = list(np.array(v0_pluto)*1000), mass = 1.303E22, name = 'Pluto' )

""" Ceres Data """ 
r0_ceres = [3.871079333790370E+08, -2.110875871409383E+08, -7.634013663169853E+07]
v0_ceres = [7.690862905958148E+00,  1.457213498229990E+01, -1.141518518761349E+00]
ceres   = body(x_vec = list(np.array(r0_ceres)*1000), v_vec = list(np.array(v0_ceres)*1000), mass = 9.3835E20, name = 'Ceres' )

""" Pallas Data """ 
r0_pallas = [3.702265427955398E+08, -3.007026255072411E+08,  1.870914361490573E+08]
v0_pallas = [9.055318351074920E+00,  8.625380511664043E+00, -6.416223865211315E+00]
pallas    = body(x_vec = list(np.array(r0_pallas )*1000), v_vec = list(np.array(v0_pallas )*1000), mass =2.04E20, name = 'pallas' )

""" Vesta Data """ 
r0_vesta = [-2.051039161971186E+08,  3.085987447832856E+08,  1.415426299976318E+07]
v0_vesta = [-1.439494636982920E+01, -1.136422342368259E+01,  2.117586193716454E+00]
vesta    = body(x_vec = list(np.array(r0_vesta)*1000), v_vec = list(np.array(v0_vesta)*1000), mass = 2.59076E20, name = 'vesta  ' )

""" Hygiea Data """ 
r0_hygiea = [-1.338532409783885E+08,  4.818048692240935E+08,  3.402184744369149E+06]
v0_hygiea = [-1.476489966085236E+01, -5.535357610703704E+00, -1.049938695172724E+00]
hygiea   = body(x_vec = list(np.array(r0_hygiea)*1000), v_vec = list(np.array(v0_hygiea)*1000), mass = 8.32E19, name = 'hygiea' )



start_time = time.time()

""" Initialize the bodies array with the above celestial body objects """ 
bodies1 = [sun,uranus,jupiter, saturn, neptune, earth, mars, mercury, venus, pluto, ceres, pallas, vesta, hygiea]

""" Empty array that will get populated with simulations for different values of G """ 
simulations = []

""" Perform the simulation for given amount of time, for different values of G """ 
for i in range(0,len(G_vals)) : 
    s_t = sim(bodies1)
    s_t.set_eqn(nbody_solve)
    s_t.runsim(t_end, d_t,G_used[i])
    s_t.plot_orbit()
    simulations.append(s_t)


"""
xU = simulation1.x_vals[7]
yU = simulation1.y_vals[7]
zU = simulation1.z_vals[7]
"""

xU_vals = []
yU_vals = []
zU_vals = []

""" Position coordinates of Uranus with respect to the Sun """ 
for i in range(0,len(G_vals)) : 
    xU_vals.append(np.subtract(simulations[i].x_vals[1], simulations[i].x_vals[0] ))
    yU_vals.append(np.subtract(simulations[i].y_vals[1], simulations[i].y_vals[0] ))
    zU_vals.append(np.subtract(simulations[i].z_vals[1], simulations[i].z_vals[0] ))


""" Uranus Data from Horizons  """
data = pd.read_csv("U_Data_1Day.csv", sep=r'\s*,\s*', engine = 'python')
r_vals_re = []
for i in range(len(data['rx'])) : 
    r = (np.sqrt(((data['rx'][i])**2) + ((data['ry'][i])**2) + ((data['rz'][i])**2) ))
    r_vals_re.append(r*km)
    
x_vals_re = []
y_vals_re = []
z_vals_re = []
for i in range(0,len(data['rx'])) : 
    x = (data['rx'][i]*km)
    y = (data['ry'][i]*km)
    z = (data['rz'][i]*km)
    x_vals_re.append(x)
    y_vals_re.append(y)
    z_vals_re.append(z)


""" Initialize array for the percentage error values """    
diff_1re = np.zeros((len(G_vals),len(xU_vals[0])))

""" Percentage error is calculated as magnitude of the difference vector between HORIZONS data and generated data """
for i in range(0,len(G_vals)) : 
        for j in range(0,len(xU_vals[0])) :
            diffmag_1re = (np.sqrt((xU_vals[i][j]-x_vals_re[j])**2 + (yU_vals[i][j]-y_vals_re[j])**2 + (zU_vals[i][j]-z_vals_re[j])**2))
            r_diff_1re = ((diffmag_1re)/(np.sqrt((x_vals_re[j])**2 + (y_vals_re[j])**2 + (z_vals_re[j])**2)))*100
            diff_1re[i][j] = r_diff_1re
     

""" To check total time taken by the program to execute """
print("total time taken : ", time.time() - start_time, "\n No of Bodies : ", len(bodies1))

""" Print the percentage error of the last value """ 
last = len(diff_1re[0]) - 1 
compare = []
for i in range(0,len(G_vals)) : 
    print(diff_1re[i][last], "----", G_vals[i], "\n")
    compare.append(diff_1re[i][last])
 
""" Create a CSV file with the percentage error values """ 
df_jvData_diff = pd.DataFrame({'Percent_error': diff_1re[0]})
df_jvData_diff.to_csv('Diff_Vectors_for_JV_data_SU.csv')

""" Create array for the time values to be plotted on X-Axis in units of "Years"  """ 
time_ax = range(0,int(365.2422*t_end) + 1)
time = list(np.array(time_ax)/365)

""" Plot the graph of percentage error versus time as a log graph """ 
plt.figure(1)
plt.plot(time, diff_1re[0], color = 'black', label = 'Model A vs Real')
plt.xlabel('Time (Years)')
plt.ylabel('%E rror')
plt.yscale('log')
plt.legend()
plt.savefig("ALL.pdf", bbox_inches='tight')








