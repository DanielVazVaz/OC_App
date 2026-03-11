import pyomo.environ as pyo
import numpy as np
import scipy.interpolate, scipy.integrate

class tank_model:
    # Understand later why this here
    T = [0, 0.155051, 0.644949, 1]
    t1 = T[1]
    t2 = T[2]
    t3 = (1)
    M1 = np.array([[1, 2*t1, 3*t1**2],
                [1, 2*t2,  3*t2**2],
                [1, 2*t3, 3*t3**2]])
    M2 = np.array([[t1, t1**2, t1**3],
                [t2, t2**2, t2**3],
                [t3, t3**2, t3**3]]) 
    OMEGA = np.linalg.inv(M1@(np.linalg.inv(M2)))
    
    def __init__(self, mode = "Simulation"):
        pass
        
    def create_model(self, ne = 9, mode = "Simulation", fixed_elements = True):
        model = pyo.ConcreteModel()

        #? SETS
        model.i = pyo.Set(initialize = [f"i{str(i+1)}" for i in range(ne)])
        model.k = pyo.Set(initialize = [f"k{str(i+1)}" for i in range(3)])
        model.j = pyo.SetOf(model.k)

        #? PARAMETERS
        k_valve_in  = 0.1
        k_valve     = 0.1 
        TIME_FINAL  = 100
        A           = 1
        L_init      = 0.06
        L_UB        = 1
        L_LB        = 0.05
        L_target    = 0.65
        w_in_UB     = 1
        w_in_LB     = 0.25
        w_UB        = 1
        w_LB        = 0.25
        OMEGA_dict = {}
        for n,j in enumerate(model.j):
            for nn,k in enumerate(model.k):
                OMEGA_dict[j,k] = tank_model.OMEGA[n,nn]
        model.OMEGA = pyo.Param(model.j, model.k, initialize = OMEGA_dict)
        weight_radau = {"k1": 0.752806/2, "k2":1.02497/2, "k3":0.222222/2}

        #? VARIABLES
        model.dLdt      = pyo.Var(model.i, model.k)
        model.F         = pyo.Var(model.i, model.k)
        model.F_in      = pyo.Var(model.i, model.k)
        model.h         = pyo.Var(model.i)
        model.L         = pyo.Var(model.i, model.k)
        model.L0        = pyo.Var(model.i)
        model.w_in      = pyo.Var(model.i, model.k)
        model.w         = pyo.Var(model.i, model.k)
        model.z         = pyo.Var()

        #? EQUATIONS
        def state_discretized(m, i, k):
            return m.L[i,k] == m.L0[i] + m.h[i]*sum(model.OMEGA[k,j]*m.dLdt[i,j] for j in m.k)
        model.eq1 = pyo.Constraint(model.i, model.k, rule = state_discretized)

        def state_derivative(m, i, k):
            return m.dLdt[i,k] == 1/A*(m.F_in[i,k] - m.F[i,k])
        model.eq2 = pyo.Constraint(model.i, model.k, rule = state_derivative)

        def valve_inflow(m,i,k):
            return m.F_in[i,k] == model.w_in[i,k]*k_valve_in
        model.eq3 = pyo.Constraint(model.i, model.k, rule = valve_inflow)

        def valve_outflow(m,i,k):
            return m.F[i,k] ==model.w[i,k]*k_valve*pyo.sqrt(m.L[i,k])
        model.eq4 = pyo.Constraint(model.i, model.k, rule = valve_outflow)

        def constant_valve_setting(m, i, k):
            if k!= "k1":
                previous_position_k = [(n+1,kk) for n,kk in enumerate(m.k) if kk == k][0][0]
                previous_position_k = m.k.at(previous_position_k - 1)
                return model.w[i,k] == model.w[i, previous_position_k]
            else:
                return pyo.Constraint.Skip
        model.eq5 = pyo.Constraint(model.i, model.k, rule = constant_valve_setting)

        def constant_valve_setting_in(m, i, k):
            if k!= "k1":
                previous_position_k = [(n+1,kk) for n,kk in enumerate(m.k) if kk == k][0][0]
                previous_position_k = m.k.at(previous_position_k - 1)
                return model.w_in[i,k] == model.w_in[i, previous_position_k]
            else:
                return pyo.Constraint.Skip
        model.eq6 = pyo.Constraint(model.i, model.k, rule = constant_valve_setting_in)

        def continuity(m, i, k):
            if i != "i1":
                previous_position_i = [(n+1,kk) for n,kk in enumerate(m.i) if kk == i][0][0]
                previous_position_i = m.i.at(previous_position_i - 1)
                return m.L0[i] == m.L[previous_position_i, "k3"]
            else:
                return pyo.Constraint.Skip
        model.eq7 = pyo.Constraint(model.i, model.k, rule = continuity) 

        if mode == "Optimization":
            def objfun(m):
                return m.z == sum(m.h[i]*weight_radau[k]*(m.L[i,k] - L_target)**2 for i in m.i for k in m.k)
            model.eq8 = pyo.Constraint(rule = objfun)

            model.obj = pyo.Objective(expr = model.z)
        elif mode == "Simulation":
            model.obj = pyo.Objective(expr = 1)

        # Fixing things
        if fixed_elements:
            model.h.fix(TIME_FINAL/len(model.i))
        else:
            model.h.setub(80); model.h.setlb(5)
            def total_time(m):
                return sum(m.h[i] for i in m.i) == TIME_FINAL
            model.eq9 = pyo.Constraint(rule = total_time)
            
        model.L0["i1"].fix(L_init)
        for key in model.L:
            model.L[key].value = L_init
        
        if mode == "Simulation":
            model.w.fix(1)
            model.w_in.fix(1)

        # Bounding things
        model.L.setub(L_UB)
        model.L.setlb(L_LB)
        model.L0.setub(L_UB)
        model.L0.setlb(L_LB)
        model.w.setlb(w_LB); model.w.setub(w_UB)
        model.w_in.setlb(w_in_LB); model.w_in.setub(w_in_UB)
        
        return model
    
    def true_model(self):
        def diff_eq(L,t):
            w_in = 1
            w = 1
            k_valve_in = 0.1
            k_valve = 0.1
            F_in = w_in*k_valve_in
            F_out = w*k_valve*np.sqrt(L)
            A = 1
            return 1/A*(F_in - F_out)
        L0 = 0.06
        real_result = scipy.integrate.odeint(diff_eq, L0, np.linspace(0,100))
        return real_result
    
    def true_model_opt(self, w_values, w_in_values, x_lag):
        #! FIX THIS
        def diff_eq(L,t, w_in_vec, w_vec, x_lag):
            for element in range(len(w_in_vec)):
                if element == 0:
                    w_in = w_in_vec[0][-1]
                    w    = w_vec[0][-1]
                if t> x_lag[element][0] and t<=x_lag[element][-1]:
                    w_in = w_in_vec[element][-1]
                    w    = w_vec[element][-1]
            k_valve_in = 0.1
            k_valve = 0.1
            F_in = w_in*k_valve_in
            F_out = w*k_valve*np.sqrt(L)
            A = 1
            return 1/A*(F_in - F_out)
        L0 = 0.06
        real_result = scipy.integrate.odeint(diff_eq, L0, np.linspace(0,100), args = (w_in_values, w_values, x_lag))
        return real_result
    
def read_results(model, variable = "L"):
    K = [0, 0.155051, 0.644949, 1] # Fixed collocation points
    if variable == "L":
        solution = model.L
        solution_0 = model.L0["i1"]
    if variable == "w_in":
        solution = model.w_in
        solution_0 = model.w_in[("i1", "k1")]
    if variable == "w_out":
        solution = model.w
        solution_0 = model.w[("i1", "k1")]
    t_vector = [0]
    variable_vector = [solution_0.value]
    base_level = 0
    for n,i in enumerate(model.i):
        if i== "i1":
            base_level = 0
        else:
            summation = 0
            for ii in model.i:
                if ii == i:
                    break
                summation += model.h[ii].value
            base_level = summation 
        for nn,k in enumerate(model.k):
            time_value = base_level + model.h[i].value*K[nn+1]
            t_vector.append(time_value)
            variable_vector.append(solution[i,k].value)
    return t_vector, variable_vector
    
    
def generate_lagrange_polynomials(results, ne = 10):
    Lagrange_polynomials = []
    x_lag = []
    y_lag = []
    start = 0
    for i in range(ne):
        end = start+4
        x_lag.append(results[0][start:end])
        y_lag.append(results[1][start:end])
        Lagrange_polynomials.append(scipy.interpolate.lagrange(x_lag[-1], y_lag[-1]))
        start = end-1
    return Lagrange_polynomials, x_lag, y_lag
        