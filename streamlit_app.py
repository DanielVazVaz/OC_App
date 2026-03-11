import streamlit as st
from tank_level_class import tank_model, read_results, generate_lagrange_polynomials
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from GLOBAL import SOLVER_PYOMO


# We set the folder of the app as the current working directory, so that we can save the images there and load them again
import os
os.chdir(os.path.dirname(__file__))


## Configuration of the page
st.set_page_config(
    page_title="Tank level control",
    page_icon="🎈",
    layout = "wide"
)

# Expander with information
with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
- Small control panel
	    """
    )

    st.markdown("")
    
mode = st.selectbox("Choose a operation mode:", ["Simulation", "Optimization"])
with st.form("my form"):
    col1, col2, col3 = st.columns([1,3,3])
    with col1:
        ne = st.text_input("Number of finite elements", "10")
        fix_fe = st.selectbox("Fix finite elements length:", ["True", "False"])
    with col2:
        st.image("Picture1.png")
    with col3:
        if mode == "Simulation":
            st.latex(r"\begin{align*} \min{z} &= 1 \\ s.t. \qquad L_{ik} &= L_{0_{i}} + h_i\sum_{j=k_1}^K \Omega_{kj} dL_{ij}\quad &\forall i\in I, k\in K \\"
                 + r"dL_{ik} &=\frac{1}{A}\left(F^{in}_{ik} - F_{ik}\right) &\forall i\in I, k\in K \\"
                 + r"F^{in}_{ik} &= w^{in}_{ik}k^{in} &\forall i\in I, k\in K \\"
                 + r"F_{ik} &= w_{ik}k\sqrt{L_{ik}} &\forall i\in I, k\in K \\"
                 + r"L_{0_{i}} &= L_{i-1 |K|} &\forall i>i_1 \\"
                 + r" \end{align*}")
            
        elif mode == "Optimization":
            st.latex(r"\begin{align*} \min{z} &= \sum_{ik}^{IK} h_i \Omega_{|K|k}\left(L_{ik} - L^{SP}\right)^2 \\"
                 + r"s.t. \qquad L_{ik} &= L_{0_{i}} + h_i\sum_{j=k_1}^K \Omega_{kj} dL_{ij}\quad &\forall i\in I, k\in K \\"
                 + r"dL_{ik} &=\frac{1}{A}\left(F^{in}_{ik} - F_{ik}\right) &\forall i\in I, k\in K \\"
                 + r"F^{in}_{ik} &= w^{in}_{ik}k^{in} &\forall i\in I, k\in K \\"
                 + r"F_{ik} &= w_{ik}k\sqrt{L_{ik}} &\forall i\in I, k\in K \\"
                 + r"L_{0_{i}} &= L_{i-1 |K|} &\forall i>i_1 \\"
                 + r"w_{ik} &= w_{ik-1} &\forall i\in I, k>k_1 \\"
                 + r"w^{in}_{ik} &= w^{in}_{ik-1} &\forall i\in I, k>k_1 \\"
                 + r" \end{align*}")
    
    if fix_fe == "True":
        fix_fe = True 
    else:
        fix_fe = False
    
    submitted = st.form_submit_button("Operate")

Level_result=None
# Solving the model
if submitted:
    TANK = tank_model()
    model = TANK.create_model(mode = mode, ne = int(ne), fixed_elements=fix_fe)
    model.i.pprint()
    opt = pyo.SolverFactory(SOLVER_PYOMO)
    result_obj = opt.solve(model, tee = True)
    Level_result = read_results(model, "L")
    w_in_result  = read_results(model, "w_in")
    w_result     = read_results(model, "w_out")
    lag_poly, x_lag, _ = generate_lagrange_polynomials(Level_result, ne = int(ne))
    _, _, w_result2 = generate_lagrange_polynomials(w_result, ne = int(ne))
    _, _, w_in_result2 = generate_lagrange_polynomials(w_in_result, ne = int(ne))

fig = plt.figure(dpi = 600, figsize = (10,4))
plt.xlabel("Time [s]")
plt.ylabel("L [m]")
if Level_result:
    plt.plot(Level_result[0], Level_result[1], "o", color = "gray", mec = "k", zorder = 10)
    if mode == "Simulation":
        plt.plot(np.linspace(0, 100), TANK.true_model(), "k", label = "True model", zorder = 8)
    else:
        results = TANK.true_model_opt(w_result2, w_in_result2, x_lag)
        #print("w_in", w_in_result2)
        #print("w_out",w_result2, "\n\n")
        #print("time_lag",x_lag, "\n\n")
        plt.plot(np.linspace(0, 100), results, "k", label = "True model", zorder = 8)
    ylim = plt.ylim()
    for i in range(int(ne)):
        x_section = np.linspace(x_lag[i][0], x_lag[i][-1],100)
        plt.vlines(x_section[-1], ylim[0], ylim[1], linestyles="--", color = "red", alpha = 0.76)
        plt.text(x_section[-1], ylim[1]+0.01, f"$i_{{{str(i+1)}}}$", ha = "center", va = "bottom", color = "red")
        plt.plot(x_section, lag_poly[i](x_section))
    if mode == "Optimization":
        xlim = plt.xlim()
        plt.text(xlim[0], 0.70, "Set point", color = "green", bbox=dict(facecolor='white', alpha=1))
        plt.hlines(0.65, xlim[0], xlim[1], linestyles="--", color = "green", alpha = 0.75)
    
    

if submitted:
    legend = plt.legend(loc = "best")
    legend.get_frame().set_alpha(1)
    plt.ylim(ylim)
    plt.grid(alpha = 0.10, zorder = 4)
    plt.tight_layout()
    xlim = plt.xlim()
    plt.savefig("temp01.png")
    st.image("temp01.png")
    
fig = plt.figure(dpi = 600, figsize = (10,4))
plt.xlabel("Time [s]")
plt.ylabel("w [-]")
if Level_result:
    plt.plot(w_in_result[0], w_in_result[1], "o-", color = "black", mec = "k", zorder = 10, label = "w_in")
    plt.plot(w_result[0], w_result[1], "o-", color = "gray", mec = "k", zorder = 10, label = "w_out")
    ylim = plt.ylim()
    for i in range(int(ne)):
        x_section = np.linspace(x_lag[i][0], x_lag[i][-1],100)
        plt.vlines(x_section[-1], ylim[0], ylim[1], linestyles="--", color = "red", alpha = 0.76)

if submitted:
    legend = plt.legend(loc = "best")
    legend.get_frame().set_alpha(1)
    plt.ylim(ylim)
    
    plt.grid(alpha = 0.10, zorder = 4)
    plt.tight_layout()
    plt.xlim(xlim)
    plt.savefig("temp02.png")
    st.image("temp02.png")
    st.write(f"Value of the objective function: {model.z.value:.2f}")