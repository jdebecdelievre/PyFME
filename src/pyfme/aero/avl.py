import numpy as np
nl = np.linalg
import os
import subprocess
import sys
import pandas as pd

FORCES = ['CL', 'CD', 'CY', 'Cl', 'Cm', 'Cn']
STAB = [F+d for F in ['CL', 'CY', 'Cl', 'Cm', 'Cn'] for d in ['a', 'b', 'p', 'q', 'r']]
CONT = [F+d for F in ['CL', 'CY', 'Cl', 'Cm', 'Cn'] for d in ['d1', 'd2', 'd3']]
# Note: missing drag

class avl_run():
    def __init__(self, geom_file, num_control_surfaces, run_file="runs", path_to_avl='.'):
        self.geom_file = geom_file
        self.run_file = run_file
        self.avl = path_to_avl
        self.num_ctrl = num_control_surfaces

    def run(self, state, controls):
        """
        State is a stack of [alpha, beta, phat, qhat, rhat] horizontal vectors
        Controls is a [elevator, aileron, rudder]
        """
        if controls.ndim > 1:
            assert controls.shape[0] == state.shape[0]

        else:
            state = np.expand_dims(state, 0)
            controls = np.expand_dims(controls, 0)
        N = controls.shape[0]

        # Modify run file
        f = open(self.run_file, 'w')
        for i in range(N):
            print(state[i])
            alpha, beta, phat, qhat, rhat = state[i]
            elevator, aileron, rudder = controls[i]
            f.write(f"""

---------------------------------------------
Run case  {i+1}:   -unnamed-

alpha        ->  alpha       =   {alpha}
beta         ->  beta        =   {beta}
pb/2V        ->  pb/2V       =   {phat}
qc/2V        ->  qc/2V       =   {qhat}
rb/2V        ->  rb/2V       =   {rhat}
elevator     ->  elevator    =   {elevator}
aileron     ->  aileron    =   {aileron}
rudder     ->  rudder    =   {rudder}

alpha     =   {alpha}     deg
beta      =   {beta}     deg
pb/2V     =   {phat}
qc/2V     =   {qhat}
rb/2V     =   {rhat}
CL        =  0.310719
CDo       =   0.00000
bank      =   0.00000     deg
elevation =   0.00000     deg
heading   =   0.00000     deg
Mach      =   0.00000
velocity  =   5.00000     Lunit/Tunit
density   =   1.12500     Munit/Lunit^3
grav.acc. =   9.81000     Lunit/Tunit^2
turn_rad. =   0.00000     Lunit
load_fac. =   1.00000
X_cg      =  0.300000     Lunit
Y_cg      =   0.00000     Lunit
Z_cg      =   0.00000     Lunit
mass      =   5.00000     Munit
Ixx       =   1.00000     Munit-Lunit^2
Iyy       =   0.02000     Munit-Lunit^2
Izz       =   1.00000     Munit-Lunit^2
Ixy       =   0.00000     Munit-Lunit^2
Iyz       =   0.00000     Munit-Lunit^2
Izx       =   0.00000     Munit-Lunit^2
visc CL_a =   0.00000
visc CL_u =   0.00000
visc CM_a =   0.00000
visc CM_u =   0.00000
""")
        f.close()

        # Create bash script
        f = open('cmd_file.run', 'w')
        # f.write(f"LOAD {self.geom_file}\n")  # load geom file
        f.write(f'PLOP\ng\n\n')  # disable graphics
        f.write(f"CASE {self.run_file}\nOPER\n")
        for i in range(N):
            results_file = f"rslt_{i}.stab"
            f.write(f"{i+1}\nx\nst\n{results_file}\n")
        f.write("\n\nQUIT")
        f.close()

        # Run bash
        with open('cmd_file.run', 'r') as commands:
            avl_run = subprocess.Popen([f"{self.avl}\\avl.exe", self.geom_file],
                                       stderr=sys.stderr,
                                       stdout=open(os.devnull, 'w'),
                                       stdin=subprocess.PIPE)
            for line in commands:
                avl_run.stdin.write(line.encode('utf-8'))
            avl_run.communicate()
            avl_run.wait()

        # sort out results
        data = pd.DataFrame({k: 0.0 for k in FORCES + STAB + CONT}, index=np.arange(N))
        data['de'] = controls[:, 0]
        data['da'] = controls[:, 1]
        data['dr'] = controls[:, 2]
        data['alpha'] = state[:, 0]
        data['beta'] = state[:, 1]
        data['p'] = state[:, 2]
        data['q'] = state[:, 3]
        data['r'] = state[:, 4]

        for i in range(N):
            with open(f"rslt_{i}.stab", 'r') as f:
                lines = f.readlines()
            data.Cl[i] = float(lines[19][33:41].strip())
            data.Cm[i] = float(lines[20][33:41].strip())
            data.Cn[i] = float(lines[21][33:41].strip())

            data.CL[i] = float(lines[23][10:20].strip())
            data.CD[i] = float(lines[24][10:20].strip())
            data.CY[i] = float(lines[20][10:20].strip())

            num_ctrl = self.num_ctrl  # number of control surfaces
            data.CLa[i] = float(lines[36 + num_ctrl][24:34].strip())  # CL_a
            data.CYa[i] = float(lines[37 + num_ctrl][24:34].strip())  # CY_a
            data.Cla[i] = float(lines[38 + num_ctrl][24:34].strip())  # Cl_a
            data.Cma[i] = float(lines[39 + num_ctrl][24:34].strip())  # Cm_a
            data.Cna[i] = float(lines[40 + num_ctrl][24:34].strip())  # Cn_a
            data.CLb[i] = float(lines[36 + num_ctrl][43:54].strip())  # CL_b
            data.CYb[i] = float(lines[37 + num_ctrl][43:54].strip())  # CY_b
            data.Clb[i] = float(lines[38 + num_ctrl][43:54].strip())  # Cl_b
            data.Cmb[i] = float(lines[39 + num_ctrl][43:54].strip())  # Cm_b
            data.Cnb[i] = float(lines[40 + num_ctrl][43:54].strip())  # Cn_b

            data.CLp[i] = float(lines[44 + num_ctrl][24:34].strip())
            data.CLq[i] = float(lines[44 + num_ctrl][43:54].strip())
            data.CLr[i] = float(lines[44 + num_ctrl][65:74].strip())
            data.CYp[i] = float(lines[45 + num_ctrl][24:34].strip())
            data.CYq[i] = float(lines[45 + num_ctrl][43:54].strip())
            data.CYr[i] = float(lines[45 + num_ctrl][65:74].strip())
            data.Clp[i] = float(lines[46 + num_ctrl][24:34].strip())
            data.Clq[i] = float(lines[46 + num_ctrl][43:54].strip())
            data.Clr[i] = float(lines[44 + num_ctrl][65:74].strip())
            data.Cmp[i] = float(lines[47 + num_ctrl][24:34].strip())
            data.Cmq[i] = float(lines[47 + num_ctrl][43:54].strip())
            data.Cmr[i] = float(lines[44 + num_ctrl][65:74].strip())
            data.Cnp[i] = float(lines[48 + num_ctrl][24:34].strip())
            data.Cnq[i] = float(lines[48 + num_ctrl][43:54].strip())
            data.Cnr[i] = float(lines[48 + num_ctrl][65:74].strip())

            INI = [24,43,65]
            FIN = [34,54,74]
            for n_ctrl in range(num_ctrl):
                data['CLd'+str(n_ctrl + 1)][i] = float(lines[52 + num_ctrl]
                                                   [INI[n_ctrl]:FIN[n_ctrl]].strip())  # CL_a
                data['CYd'+str(n_ctrl + 1)][i] = float(lines[53 + num_ctrl]
                                                   [INI[n_ctrl]:FIN[n_ctrl]].strip())  # CY_a
                data['Cld'+str(n_ctrl + 1)][i] = float(lines[54 + num_ctrl]
                                                   [INI[n_ctrl]:FIN[n_ctrl]].strip())  # Cl_a
                data['Cmd'+str(n_ctrl + 1)][i] = float(lines[55 + num_ctrl]
                                                   [INI[n_ctrl]:FIN[n_ctrl]].strip())  # Cm_a
                data['Cnd'+str(n_ctrl + 1)][i] = float(lines[56 + num_ctrl]
                                                   [INI[n_ctrl]:FIN[n_ctrl]].strip())  # Cn_a

            os.remove(f"rslt_{i}.stab")
        os.remove(self.run_file)
        os.remove('cmd_file.run')
        return(data)

##### TODO : try to see if I can leave an AVL session open

# def count_control_surfaces(geomfile):
#     with open(geomfile,'r') as f:
#         for line in f:
#


if __name__ == "__main__":
    states = []
    controls = []
    for al in np.linspace(-10,20,30):
        # for de in np.linspace(-26,28,10):
        #     controls.append(np.array([de,0,0]))
        #     states.append(np.array([al,0,0,0,0]))
        # for da in np.linspace(-15, 10):
        #     controls.append(np.array([0,da,0]))
        #     states.append(np.array([al,0,0,0,0]))
        # for dr in np.linspace(-5,5,10):
        #     controls.append(np.array([0,0,dr]))
        states.append(np.array([al,0,0,0,0]))
        controls.append(np.array([0,0,0]))
    states = np.array(states)
    controls = np.array(controls)
    # states = np.array([np.arange(5)/100, np.arange(5)/600])
    # controls = np.array([np.arange(3)/3, np.arange(3)/4])
    a = avl_run(num_control_surfaces=3, geom_file='hypo', run_file='runs')
    data = a.run(states, controls)
    data.to_pickle('MeterSpanUAV.pkl')