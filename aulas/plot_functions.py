import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import widgets, FloatSlider, FloatText, VBox, HBox, GridBox
from IPython.display import display


def single_vector(half_length=20, slider_step=0.5, figure_size=7):
    '''
    Plot a single vector in a 2D Cartesian system.
    The plot area extends from -"half_length" to +"half_length" 
    along x and y axes. The size of the vector (Cartesian) components 
    are controlled by sliders having step defined by "slider_step".
    The figure size is defined by "figure_size".
    '''

    assert isinstance(half_length, (float, int)), "half_length must be a scalar"
    assert half_length > 0, "half_length must be positive"
    assert isinstance(slider_step, (float, int)), "slider_step must be a scalar"
    assert slider_step > 0, "slider_step must be positive"
    assert isinstance(figure_size, (float, int)), "figure_size must be a scalar"
    assert figure_size > 0, "figure_size must be positive"

    area = [-half_length, half_length, -half_length, half_length]

    scale = 2 * half_length

    # Set initial values
    x_origin = 0
    y_origin = 0
    x_component_init = 0.5 * half_length
    y_component_init = 0.5 * half_length

    # create Output widget to hold the figure
    out_fig = widgets.Output()

    with out_fig:
        fig, ax = plt.subplots(figsize=(figure_size,figure_size))
        # reference system
        ax.quiver(
            area[0], 0, 
            scale, 0,
            scale=scale, color='k'
        )
        ax.quiver(
            0, area[2], 
            0, scale, 
            scale=scale, color='k'
        )
        # vector
        Q = ax.quiver(
            x_origin, 
            y_origin, 
            x_component_init, 
            y_component_init,
            scale=scale, 
            color="blue"
            )
        # vector component lines
        L1, = ax.plot(
            [0, x_component_init], 
            [0, 0], 
            color='b', ls='--', lw=1
        )
        L2, = ax.plot(
            [0, 0], 
            [0, y_component_init], 
            color='b', ls='--', lw=1
        )
        L3, = ax.plot(
            [0, x_component_init], 
            [y_component_init, y_component_init], 
            color='b', ls='--', lw=1
        )
        L4, = ax.plot(
            [x_component_init, x_component_init], 
            [0, y_component_init], 
            color='b', ls='--', lw=1
        )
        ax.set_aspect("equal")
        ax.set_xlim(area[0], area[1])
        ax.set_ylim(area[2], area[3])
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.grid()

    # Create the sliders
    x_component_slider = FloatSlider(
        min=area[0], max=area[1], 
        step=slider_step, value=x_component_init, 
        description='$v_{x}$'
        )
    y_component_slider = FloatSlider(
        min=area[2], max=area[3], 
        step=slider_step, value=y_component_init, 
        description='$v_{y}$'
        )

    # Pack sliders in a vertical box
    sliders = HBox([x_component_slider, y_component_slider])

    # update function
    def update(
        x_component = 0.5 * half_length, 
        y_component = 0.5 * half_length
        ):
        U = x_component
        V = y_component
        # update vector
        Q.set_UVC(U, V)
        # update vector component lines
        L1.set_xdata([0, U])
        L2.set_ydata([0, V])
        L3.set_data([0, U], [V, V])
        L4.set_data([U, U], [0, V])
        fig.canvas.draw_idle()

    # Update sliders
    widgets.interactive_output(update, {
        "x_component": x_component_slider,
        "y_component": y_component_slider
        })

    ui = HBox([out_fig, sliders])
    
    display(ui)


def scalar_product(
    half_length=20, slider_step=0.5, figure_size=7
    ):
    '''
    Plot two vectors v1 and v2 in a 2D Cartesian system.
    The plot area extends from -"half_length" to +"half_length" 
    along x and y axes. The norm and angle of each vector 
    are controlled by sliders having step defined by "slider_step".
    The figure size is defined by "figure_size".
    '''

    assert isinstance(half_length, (float, int)), "half_length must be a scalar"
    assert half_length > 0, "half_length must be positive"
    assert isinstance(slider_step, (float, int)), "slider_step must be a scalar"
    assert slider_step > 0, "slider_step must be positive"
    assert isinstance(figure_size, (float, int)), "figure_size must be a scalar"
    assert figure_size > 0, "figure_size must be positive"

    area = [-half_length, half_length, -half_length, half_length]

    scale = 2 * half_length

    # Set initial values
    v1_norm_init = 0.5 * half_length
    v2_norm_init = 0.3 * half_length
    v1_theta_init = 30
    v2_theta_init = 45

    # vectors 
    theta1 = np.deg2rad(v1_theta_init)
    theta2 = np.deg2rad(v2_theta_init)
    v1x = v1_norm_init * np.cos(theta1)
    v1y = v1_norm_init * np.sin(theta1)
    v2x = v2_norm_init * np.cos(theta2)
    v2y = v2_norm_init * np.sin(theta2)
    dot_val_init = v1x*v2x + v1y*v2y

    # create Output widget to hold the figure
    out_fig = widgets.Output()

    with out_fig:
        fig, ax = plt.subplots(figsize=(figure_size,figure_size))
        # reference system
        ax.quiver(
            area[0], 0, 
            scale, 0,
            scale=scale, color='k'
        )
        ax.quiver(
            0, area[2], 
            0, scale, 
            scale=scale, color='k'
        )

        # vectors 
        theta1 = np.deg2rad(v1_theta_init)
        theta2 = np.deg2rad(v2_theta_init)

        v1x = v1_norm_init * np.cos(theta1)
        v1y = v1_norm_init * np.sin(theta1)

        v2x = v2_norm_init * np.cos(theta2)
        v2y = v2_norm_init * np.sin(theta2)

        Q1 = ax.quiver(
            0, 0, 
            v1x, v1y, 
            scale=scale, color='b'
        )

        Q2 = ax.quiver(
            0, 0, 
            v2x, v2y, 
            scale=scale, color='r'
        )
        
        ax.set_aspect("equal")
        ax.set_xlim(area[0], area[1])
        ax.set_ylim(area[2], area[3])
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.grid()

    # Create the sliders
    v1_norm_description = "$\| \mathbf{v}_{1} \|$"
    v2_norm_description = "$\| \mathbf{v}_{2} \|$"
    v1_theta_description = "$\\theta_{1}$"
    v2_theta_description = "$\\theta_{2}$"
    v1_dot_v2_description = (
        "$\mathbf{v}_{1}^{\\top}\mathbf{v}_{2} = $"
        )

    v1_norm_widget = FloatSlider(
        min=-half_length, max=half_length, step=slider_step, 
        value=v1_norm_init, description=v1_norm_description)
    v2_norm_widget = FloatSlider(
        min=-half_length, max=half_length, step=slider_step, 
        value=v2_norm_init, description=v2_norm_description)
    v1_theta_widget = FloatSlider(
        min=0, max=360, step=slider_step, 
        value=v1_theta_init, description=v1_theta_description)
    v2_theta_widget = FloatSlider(
        min=0, max=360, step=slider_step, 
        value=v2_theta_init, description=v2_theta_description)
    v1_dot_v2_widget = FloatText(
        value=dot_val_init,
        description=v1_dot_v2_description,
        disabled=True
        )

    # Pack sliders in a vertical box
    sliders = HBox([
        v1_norm_widget, v2_norm_widget, 
        v1_theta_widget, v2_theta_widget, 
        v1_dot_v2_widget
        ])

    # update function
    def update(
            v1_norm = 0.5 * half_length,
            v2_norm = 0.3 * half_length,
            v1_theta = 30,
            v2_theta = 45
            ):

            # vectors 
            theta1 = np.deg2rad(v1_theta)
            theta2 = np.deg2rad(v2_theta)

            v1x = v1_norm * np.cos(theta1)
            v1y = v1_norm * np.sin(theta1)

            v2x = v2_norm * np.cos(theta2)
            v2y = v2_norm * np.sin(theta2)

            # update vectors
            Q1.set_UVC(U=v1x, V=v1y)
            Q2.set_UVC(U=v2x, V=v2y)

            # update dot product
            dot_val = v1x*v2x + v1y*v2y
            v1_dot_v2_widget.value = f"{dot_val:.3f}"

            fig.canvas.draw_idle()

    # Update sliders
    widgets.interactive_output(update, {
        "v1_norm": v1_norm_widget,
        "v2_norm": v2_norm_widget,
        "v1_theta": v1_theta_widget,
        "v2_theta": v2_theta_widget
    })

    ui = HBox([out_fig, sliders])
    display(ui)


def vector_product( half_length=20, slider_step=0.5, figure_size=7 ): 
    '''
    Plot three vectors v1, v2 and v3 in a 3D Cartesian system, where v3 = v1 x v2. 
    The plot area extends from -"half_length" to +"half_length" along x, y and z axes. 
    The norm, declination and inclination of vectors v1 and v2 are controlled by 
    sliders having step defined by "slider_step". The figure size is defined by 
    "figure_size".
    '''

    assert isinstance(half_length, (float, int)), "half_length must be a scalar"
    assert half_length > 0, "half_length must be positive" 
    assert isinstance(slider_step, (float, int)), "slider_step must be a scalar" 
    assert slider_step > 0, "slider_step must be positive" 
    assert isinstance(figure_size, (float, int)), "figure_size must be a scalar" 
    assert figure_size > 0, "figure_size must be positive" 

    area = [
        -half_length, half_length, 
        -half_length, half_length, 
        -half_length, half_length
        ]

    scale = 2 * half_length 

    # Set initial values 
    v1_norm_init = 0.5 * half_length 
    v2_norm_init = 0.3 * half_length 
    v1_dec_init = 30 
    v2_dec_init = 45 
    v1_inc_init = 0 
    v2_inc_init = 0 

    # Compute Cartesian components 
    def spherical_to_Cartesian(norm, inc, dec): 
        D = np.deg2rad(dec) 
        I = np.deg2rad(inc) 
        vx = norm * np.cos(I) * np.cos(D) 
        vy = norm * np.cos(I) * np.sin(D) 
        vz = norm * np.sin(I) 
        return vx, vy, vz 

    v1x, v1y, v1z = spherical_to_Cartesian(
        v1_norm_init, v1_inc_init, v1_dec_init
        ) 
    v2x, v2y, v2z = spherical_to_Cartesian(
        v2_norm_init, v2_inc_init, v2_dec_init
        ) 

    # Compute the vector product 
    v3x, v3y, v3z = np.cross([v1x, v1y, v1z], [v2x, v2y, v2z]) 
    v3_norm_init = np.linalg.norm([v3x, v3y, v3z]) 

    # create Output widget to hold the figure 
    out_fig = widgets.Output() 

    # Initialize quiver variables in outer scope
    Q1 = Q2 = Q3 = None

    with out_fig: 
        fig = plt.figure(figsize=(figure_size, figure_size)) 
        ax = fig.add_subplot(111, projection='3d') 

        # vectors 
        Q1 = ax.quiver( 0, 0, 0, v1x, v1y, v1z, lw=3, arrow_length_ratio=0.05, color='b' ) 
        Q2 = ax.quiver( 0, 0, 0, v2x, v2y, v2z, lw=3, arrow_length_ratio=0.05, color='r' ) 
        Q3 = ax.quiver( 0, 0, 0, v3x, v3y, v3z, lw=3, arrow_length_ratio=0.05, color='g' ) 

        # reference system 
        ax.quiver( area[0], 0, 0, scale, 0, 0, arrow_length_ratio=0.05, color='k' ) 
        ax.quiver( 0, area[2], 0, 0, scale, 0, arrow_length_ratio=0.05, color='k' ) 
        ax.quiver( 0, 0, area[4], 0, 0, scale, arrow_length_ratio=0.05, color='k' ) 

        ax.set_aspect("equal") 
        ax.set_xlim(area[0], area[1]) 
        ax.set_ylim(area[2], area[3]) 
        ax.set_zlim(area[4], area[5]) 
        ax.set_xlabel('x', fontsize=14) 
        ax.set_ylabel('y', fontsize=14) 
        ax.set_zlabel('z', fontsize=14) 
        ax.grid() 

    # Create the sliders 
    
    v1_norm_description = "$\| \mathbf{v}_{1} \|$" 
    v2_norm_description = "$\| \mathbf{v}_{2} \|$" 
    v3_norm_description = "$\| \mathbf{v}_{3} \|$" 
    v1_dec_description = "$D_{1}$" 
    v2_dec_description = "$D_{2}$" 
    v1_inc_description = "$I_{1}$" 
    v2_inc_description = "$I_{2}$" 

    v1_norm_widget = FloatSlider( 
        min=-half_length, max=half_length, step=slider_step, 
        value=v1_norm_init, description=v1_norm_description
        ) 
    v2_norm_widget = FloatSlider( 
        min=-half_length, max=half_length, step=slider_step, 
        value=v2_norm_init, description=v2_norm_description
        ) 
    v1_dec_widget = FloatSlider( 
        min=0, max=360, step=slider_step, 
        value=v1_dec_init, description=v1_dec_description
        ) 
    v2_dec_widget = FloatSlider( 
        min=0, max=360, step=slider_step, 
        value=v2_dec_init, description=v2_dec_description
        ) 
    v1_inc_widget = FloatSlider( 
        min=-90, max=90, step=slider_step, 
        value=v1_inc_init, description=v1_inc_description
        ) 
    v2_inc_widget = FloatSlider( 
        min=-90, max=90, step=slider_step, 
        value=v2_inc_init, description=v2_inc_description
        ) 
    v3_norm_widget = FloatText( 
        value=f"{v3_norm_init:.3f}" , description=v3_norm_description, disabled=True 
        ) 

    # Pack sliders in a box 
    sliders1 = HBox([v1_norm_widget, v1_dec_widget, v1_inc_widget])
    sliders2 = HBox([v2_norm_widget, v2_dec_widget, v2_inc_widget])
    sliders = VBox([sliders1, sliders2, v3_norm_widget])

    # update function 
    def update(change=None):
        nonlocal Q1, Q2, Q3
        with out_fig:

            # vectors 
            v1x, v1y, v1z = spherical_to_Cartesian( 
                v1_norm_widget.value, v1_inc_widget.value, v1_dec_widget.value 
                ) 
            v2x, v2y, v2z = spherical_to_Cartesian( 
                v2_norm_widget.value, v2_inc_widget.value, v2_dec_widget.value 
                ) 

            # Compute the vector product 
            v3x, v3y, v3z = np.cross([v1x, v1y, v1z], [v2x, v2y, v2z]) 
            v3_norm = np.linalg.norm([v3x, v3y, v3z]) 

            # Efficiently remove old quivers
            for Q in [Q1, Q2, Q3]:
                if Q is not None:
                    Q.remove()

            # Draw new quivers
            Q1 = ax.quiver(0,0,0, v1x,v1y,v1z, lw=3, arrow_length_ratio=0.1, color='b')
            Q2 = ax.quiver(0,0,0, v2x,v2y,v2z, lw=3, arrow_length_ratio=0.1, color='r')
            Q3 = ax.quiver(0,0,0, v3x,v3y,v3z, lw=3, arrow_length_ratio=0.1, color='g')

            # update the vector product 
            v3_norm_widget.value = f"{v3_norm:.3f}" 

            fig.canvas.draw_idle() 

    # Attach observers to sliders
    for w in [
        v1_norm_widget, v2_norm_widget, 
        v1_dec_widget, v2_dec_widget, 
        v1_inc_widget, v2_inc_widget
        ]:
        w.observe(update, names='value')

    # Initial draw
    update()

    # Display figure and sliders
    ui = HBox([out_fig, sliders])
    display(ui)


def R1(psi):
    '''
    Rotation matrix around x axis.
    '''

    psi_rad = np.deg2rad(psi)
    cos = np.cos(psi_rad)
    sin = np.sin(psi_rad)

    R1 = np.array([
        [   1,    0,    0], 
        [   0,  cos,  sin], 
        [   0, -sin,  cos]])

    return R1


def R2(psi):
    '''
    Rotation matrix around y axis.
    '''

    psi_rad = np.deg2rad(psi)
    cos = np.cos(psi_rad)
    sin = np.sin(psi_rad)

    R2 = np.array([
        [  cos,   0, -sin], 
        [    0,   1,    0], 
        [  sin,   0,  cos]])

    return R2


def R3(psi):
    '''
    Rotation matrix around z axis.
    '''

    psi_rad = np.deg2rad(psi)
    cos = np.cos(psi_rad)
    sin = np.sin(psi_rad)

    R3 = np.array([
        [  cos,  sin,    0], 
        [ -sin,  cos,    0], 
        [    0,    0,    1]])

    return R3


def rotation_coordinates( half_length=20, slider_step=0.5, figure_size=7 ): 
    '''
    Plot two vectors v1 and v2 in the same 3D Cartesian system, where v2 = R v1, with 
    R being a resultant rotation matrix R = R1 R2 R3, defined in terms of three rotation 
    matrices R1, R2 e R3, by Euler angles e1, e2 and e3, respectively. 
    The plot area extends from -"half_length" to +"half_length" along x, y and z axes. 
    The norm, declination and inclination of vectors v1 and v2 are controlled by 
    sliders having step defined by "slider_step". The figure size is defined by 
    "figure_size".
    '''

    assert isinstance(half_length, (float, int)), "half_length must be a scalar"
    assert half_length > 0, "half_length must be positive" 
    assert isinstance(slider_step, (float, int)), "slider_step must be a scalar" 
    assert slider_step > 0, "slider_step must be positive" 
    assert isinstance(figure_size, (float, int)), "figure_size must be a scalar" 
    assert figure_size > 0, "figure_size must be positive" 

    area = [
        -half_length, half_length, 
        -half_length, half_length, 
        -half_length, half_length
        ]

    scale = 2 * half_length 

    # Set initial values of vector v1
    v1_norm_init = 0.5 * half_length 
    v1_dec_init = 30 
    v1_inc_init = 0 

    # Set initial rotation angles 
    e1_init = 0
    e2_init = 0
    e3_init = 0

    # Compute Cartesian components 
    def spherical_to_Cartesian(norm, inc, dec): 
        D = np.deg2rad(dec) 
        I = np.deg2rad(inc) 
        vx = norm * np.cos(I) * np.cos(D) 
        vy = norm * np.cos(I) * np.sin(D) 
        vz = norm * np.sin(I) 
        return np.array([vx, vy, vz])

    # Define vector v1 in terms of initial values
    v1 = spherical_to_Cartesian(
        v1_norm_init, v1_inc_init, v1_dec_init
        ) 

    # Compute the rotation matrix
    R = R1(e1_init) @ R2(e2_init) @ R3(e3_init)

    # Define vector v2
    v2 = R @ v1

    # create Output widget to hold the figure 
    out_fig = widgets.Output() 

    # Initialize quiver variables in outer scope
    Q1 = Q2 = None

    with out_fig: 
        fig = plt.figure(figsize=(figure_size, figure_size)) 
        ax = fig.add_subplot(111, projection='3d') 

        # vectors 
        Q1 = ax.quiver( 0, 0, 0, v1[0], v1[1], v1[2], lw=3, arrow_length_ratio=0.05, color='b' ) 
        Q2 = ax.quiver( 0, 0, 0, v2[0], v2[1], v2[2], lw=3, arrow_length_ratio=0.05, color='r' ) 

        # reference system 
        ax.quiver( area[0], 0, 0, scale, 0, 0, arrow_length_ratio=0.05, color='k' ) 
        ax.quiver( 0, area[2], 0, 0, scale, 0, arrow_length_ratio=0.05, color='k' ) 
        ax.quiver( 0, 0, area[4], 0, 0, scale, arrow_length_ratio=0.05, color='k' ) 

        ax.set_aspect("equal") 
        ax.set_xlim(area[0], area[1]) 
        ax.set_ylim(area[2], area[3]) 
        ax.set_zlim(area[4], area[5]) 
        ax.set_xlabel('x', fontsize=14) 
        ax.set_ylabel('y', fontsize=14) 
        ax.set_zlabel('z', fontsize=14) 
        ax.grid() 

    # Create the sliders 
    
    v1_norm_description = "$\| \mathbf{v}_{1} \|$" 
    v1_dec_description = "$D_{1}$" 
    v1_inc_description = "$I_{1}$" 
    e1_description = "$\epsilon_{1}$"
    e2_description = "$\epsilon_{2}$"
    e3_description = "$\epsilon_{3}$"

    v1_norm_widget = FloatSlider( 
        min=-half_length, max=half_length, step=slider_step, 
        value=v1_norm_init, description=v1_norm_description
        ) 
    v1_dec_widget = FloatSlider( 
        min=0, max=360, step=slider_step, 
        value=v1_dec_init, description=v1_dec_description
        ) 
    v1_inc_widget = FloatSlider( 
        min=-90, max=90, step=slider_step, 
        value=v1_inc_init, description=v1_inc_description
        )
    e1_widget = FloatSlider( 
        min=-180, max=180, step=slider_step, 
        value=e1_init, description=e1_description
        )
    e2_widget = FloatSlider( 
        min=-180, max=180, step=slider_step, 
        value=e2_init, description=e2_description
        )
    e3_widget = FloatSlider( 
        min=-180, max=180, step=slider_step, 
        value=e3_init, description=e3_description
        )

    # Pack sliders in a box  
    sliders1 = HBox([v1_norm_widget, v1_dec_widget, v1_inc_widget])
    sliders2 = HBox([e1_widget, e2_widget, e3_widget])
    sliders = VBox([sliders1, sliders2])

    # update function 
    def update(change=None):
        nonlocal Q1, Q2
        with out_fig:

            # vectors 
            v1 = spherical_to_Cartesian( 
                v1_norm_widget.value, v1_inc_widget.value, v1_dec_widget.value 
                )

            # Compute the rotation matrix
            R = R1(e1_widget.value) @ R2(e2_widget.value) @ R3(e3_widget.value)

            # Define vector v2
            v2 = R @ v1

            # Efficiently remove old quivers
            for Q in [Q1, Q2]:
                if Q is not None:
                    Q.remove()

            # Draw new quivers
            Q1 = ax.quiver(0,0,0, v1[0],v1[1],v1[2], lw=3, arrow_length_ratio=0.1, color='b')
            Q2 = ax.quiver(0,0,0, v2[0],v2[1],v2[2], lw=3, arrow_length_ratio=0.1, color='r')

            fig.canvas.draw_idle() 

    # Attach observers to sliders
    for w in [
        v1_norm_widget, v1_dec_widget, v1_inc_widget, 
        e1_widget, e2_widget, e3_widget
        ]:
        w.observe(update, names='value')

    # Initial draw
    update()

    # Display figure and sliders
    ui = VBox([out_fig, sliders])
    display(ui)


def rotation_base( half_length=20, slider_step=0.5, figure_size=7 ): 
    '''
    Plot one vector v in two different 3D Cartesian systems. 
    The axes of the second Cartesian system are defined by the columns of a rotation
    matrix R, which is defined in terms of three rotation matrices R1, R2 e R3, 
    by Euler angles e1, e2 and e3, respectively. 
    The plot area extends from -"half_length" to +"half_length" along x, y and z axes. 
    The norm, declination and inclination of vectors v1 and v2 are controlled by 
    sliders having step defined by "slider_step". The figure size is defined by 
    "figure_size".
    '''

    assert isinstance(half_length, (float, int)), "half_length must be a scalar"
    assert half_length > 0, "half_length must be positive" 
    assert isinstance(slider_step, (float, int)), "slider_step must be a scalar" 
    assert slider_step > 0, "slider_step must be positive" 
    assert isinstance(figure_size, (float, int)), "figure_size must be a scalar" 
    assert figure_size > 0, "figure_size must be positive" 

    area = [
        -half_length, half_length, 
        -half_length, half_length, 
        -half_length, half_length
        ]

    scale = 2 * half_length 

    # Set the coordinates of v in the reference coordinate system 1
    v1_norm_init = 0.5 * half_length 
    v1_dec_init = 30 
    v1_inc_init = 0 

    # Set initial rotation angles 
    e1_init = 0
    e2_init = 0
    e3_init = 0

    # Compute Cartesian components 
    def spherical_to_Cartesian(norm, inc, dec): 
        D = np.deg2rad(dec) 
        I = np.deg2rad(inc) 
        vx = norm * np.cos(I) * np.cos(D) 
        vy = norm * np.cos(I) * np.sin(D) 
        vz = norm * np.sin(I) 
        return np.array([vx, vy, vz])

    # Define the Cartesian components of v in the 
    # reference Cartesian system 1
    v1 = spherical_to_Cartesian(v1_norm_init, v1_inc_init, v1_dec_init) 

    # Compute the rotation matrix defining the axes of
    # the Cartesian system 2
    R = R1(e1_init) @ R2(e2_init) @ R3(e3_init)

    # Define the components of v in the Cartesian system 2
    v2 = R.T @ v1

    # create Output widget to hold the figure 
    out_fig = widgets.Output() 

    # Initialize quiver variables in outer scope
    Qv = None    # vector v
    Qv2x = None  # x-component of v in system 2
    Qv2y = None  # y-component of v in system 2
    Qv2z = None  # z-component of v in system 2
    Q2x = None   # x-axis of system 2
    Q2y = None   # y-axis of system 2
    Q2z = None   # z-axis of system 2

    with out_fig: 
        fig = plt.figure(figsize=(figure_size, figure_size)) 
        ax = fig.add_subplot(111, projection='3d') 

        # Components of v in the system 2
        Rj = v2[0] * R[:,0]
        Qv2x = ax.quiver( 0, 0, 0, Rj[0], Rj[1], Rj[2], lw=3, arrow_length_ratio=0.05, color='r', zorder=0 ) 
        Rj = v2[1] * R[:,1]
        Qv2y = ax.quiver( 0, 0, 0, Rj[0], Rj[1], Rj[2], lw=3, arrow_length_ratio=0.05, color='r', zorder=0 ) 
        Rj = v2[2] * R[:,2]
        Qv2z = ax.quiver( 0, 0, 0, Rj[0], Rj[1], Rj[2], lw=3, arrow_length_ratio=0.05, color='r', zorder=0 ) 

        # Axes of system 2
        Rj = 0.8 * half_length * R[:,0]
        Q2x = ax.quiver( 
            -Rj[0], -Rj[1], -Rj[2], 2*Rj[0], 2*Rj[1], 2*Rj[2], 
            lw=3, arrow_length_ratio=0.05, color='k', alpha=0.7, zorder=3 )
        Rj = 0.8 * half_length * R[:,1]
        Q2y = ax.quiver( 
            -Rj[0], -Rj[1], -Rj[2], 2*Rj[0], 2*Rj[1], 2*Rj[2], 
            lw=3, arrow_length_ratio=0.05, color='k', alpha=0.7, zorder=3 )
        Rj = 0.8 * half_length * R[:,2]
        Q2z = ax.quiver( 
            -Rj[0], -Rj[1], -Rj[2], 2*Rj[0], 2*Rj[1], 2*Rj[2], 
            lw=3, arrow_length_ratio=0.05, color='k', alpha=0.7, zorder=3 ) 

        # Vector v
        Qv = ax.quiver( 0, 0, 0, v1[0], v1[1], v1[2], lw=3, arrow_length_ratio=0.05, color='r' ) 

        # reference system 
        ax.quiver( area[0], 0, 0, scale, 0, 0, arrow_length_ratio=0.05, color=3*(0.7,), alpha=0.7, zorder=5 ) 
        ax.quiver( 0, area[2], 0, 0, scale, 0, arrow_length_ratio=0.05, color=3*(0.7,), alpha=0.7, zorder=5 ) 
        ax.quiver( 0, 0, area[4], 0, 0, scale, arrow_length_ratio=0.05, color=3*(0.7,), alpha=0.7, zorder=5 ) 

        ax.set_aspect("equal") 
        ax.set_xlim(area[0], area[1]) 
        ax.set_ylim(area[2], area[3]) 
        ax.set_zlim(area[4], area[5]) 
        ax.set_xlabel('x', fontsize=14) 
        ax.set_ylabel('y', fontsize=14) 
        ax.set_zlabel('z', fontsize=14) 
        ax.grid() 

    # Create the sliders 
    
    v1_norm_description = "$\| \mathbf{v}_{1} \|$" 
    v1_dec_description = "$D_{1}$" 
    v1_inc_description = "$I_{1}$" 
    e1_description = "$\epsilon_{1}$"
    e2_description = "$\epsilon_{2}$"
    e3_description = "$\epsilon_{3}$"

    v1_norm_widget = FloatSlider( 
        min=-half_length, max=half_length, step=slider_step, 
        value=v1_norm_init, description=v1_norm_description
        ) 
    v1_dec_widget = FloatSlider( 
        min=0, max=360, step=slider_step, 
        value=v1_dec_init, description=v1_dec_description
        ) 
    v1_inc_widget = FloatSlider( 
        min=-90, max=90, step=slider_step, 
        value=v1_inc_init, description=v1_inc_description
        )
    e1_widget = FloatSlider( 
        min=-180, max=180, step=slider_step, 
        value=e1_init, description=e1_description
        )
    e2_widget = FloatSlider( 
        min=-180, max=180, step=slider_step, 
        value=e2_init, description=e2_description
        )
    e3_widget = FloatSlider( 
        min=-180, max=180, step=slider_step, 
        value=e3_init, description=e3_description
        )

    # Pack sliders in a box  
    sliders1 = HBox([v1_norm_widget, v1_dec_widget, v1_inc_widget])
    sliders2 = HBox([e1_widget, e2_widget, e3_widget])
    sliders = VBox([sliders1, sliders2])

    # update function 
    def update(change=None):
        nonlocal Qv, Q2x, Q2y, Q2z, Qv2x, Qv2y, Qv2z
        with out_fig:

            # Components of vector v in the reference system 1
            v1 = spherical_to_Cartesian( 
                v1_norm_widget.value, v1_inc_widget.value, v1_dec_widget.value 
                )

            # Rotation matrix
            R = R1(e1_widget.value) @ R2(e2_widget.value) @ R3(e3_widget.value)

            # Components of vector v in the reference system 2
            v2 = R.T @ v1

            # Efficiently remove old quivers
            for Q in [Qv, Q2x, Q2y, Q2z, Qv2x, Qv2y, Qv2z]:
                if Q is not None:
                    Q.remove()

            # Draw new quivers for components of v in the system 2
            Rj = v2[0] * R[:,0]
            Qv2x = ax.quiver( 0, 0, 0, Rj[0], Rj[1], Rj[2], lw=3, arrow_length_ratio=0.05, color='r', zorder=0 ) 
            Rj = v2[1] * R[:,1]
            Qv2y = ax.quiver( 0, 0, 0, Rj[0], Rj[1], Rj[2], lw=3, arrow_length_ratio=0.05, color='r', zorder=0 ) 
            Rj = v2[2] * R[:,2]
            Qv2z = ax.quiver( 0, 0, 0, Rj[0], Rj[1], Rj[2], lw=3, arrow_length_ratio=0.05, color='r', zorder=0 ) 

            # Draw new quivers for axes of system 2
            Rj = 0.8 * half_length * R[:,0]
            Q2x = ax.quiver( 
                -Rj[0], -Rj[1], -Rj[2], 2*Rj[0], 2*Rj[1], 2*Rj[2], 
                lw=3, arrow_length_ratio=0.05, color='k', alpha=0.7, zorder=3 )
            Rj = 0.8 * half_length * R[:,1]
            Q2y = ax.quiver( 
                -Rj[0], -Rj[1], -Rj[2], 2*Rj[0], 2*Rj[1], 2*Rj[2], 
                lw=3, arrow_length_ratio=0.05, color='k', alpha=0.7, zorder=3 )
            Rj = 0.8 * half_length * R[:,2]
            Q2z = ax.quiver( 
                -Rj[0], -Rj[1], -Rj[2], 2*Rj[0], 2*Rj[1], 2*Rj[2], 
                lw=3, arrow_length_ratio=0.05, color='k', alpha=0.7, zorder=3 ) 

            # Draw new quiver for vector v
            Qv = ax.quiver( 0, 0, 0, v1[0], v1[1], v1[2], lw=3, arrow_length_ratio=0.05, color='r' ) 

            fig.canvas.draw_idle() 

    # Attach observers to sliders
    for w in [
        v1_norm_widget, v1_dec_widget, v1_inc_widget, 
        e1_widget, e2_widget, e3_widget
        ]:
        w.observe(update, names='value')

    # Initial draw
    update()

    # Display figure and sliders
    ui = VBox([out_fig, sliders])
    display(ui)


def sphere_gravity_potential_reference(half_length=20, slider_step=0.5, figure_size=7):
    '''
    Plot a sphere and its gravitational potential in a 2D Cartesian system.
    The plot area extends from -"half_length" to +"half_length" 
    along x and y axes. The size of the vector (Cartesian) components 
    are controlled by sliders having step defined by "slider_step".
    The figure size is defined by "figure_size".
    '''

    assert isinstance(half_length, (float, int)), "half_length must be a scalar"
    assert half_length > 0, "half_length must be positive"
    assert isinstance(slider_step, (float, int)), "slider_step must be a scalar"
    assert slider_step > 0, "slider_step must be positive"
    assert isinstance(figure_size, (float, int)), "figure_size must be a scalar"
    assert figure_size > 0, "figure_size must be positive"

    area = [-half_length, half_length, -half_length, half_length]

    scale = 2 * half_length

    # Set computation points
    npoints = 100
    shape = (npoints, npoints)
    coordinates = np.linspace(-half_length, half_length, npoints)
    X = np.broadcast_to(coordinates, shape).T
    Z = np.broadcast_to(coordinates, shape)
    R = np.sqrt(X**2 + Z**2)

    # Set the initial radius
    R0_init = 0.3 * half_length
    # Set the initial density
    rho_init = 2

    # create Output widget to hold the figure
    out_fig = widgets.Output()

    with out_fig:
        fig, ax = plt.subplots(figsize=(figure_size,figure_size))

        ax.set_aspect("equal")
        ax.set_xlim(area[0], area[1])
        ax.set_ylim(area[2], area[3])

        # compute the gravitational potential
        potential = gravitational_potential(R0_init, rho_init, R)

        # plot the gravitational potential
        potential_plot = ax.pcolormesh(
            X, Z, potential, 
            shading='nearest', cmap='jet', vmin=0, vmax=rho_init*2*np.pi*(R0_init**2)
            )

       # create a divider for the existing axes
        divider = make_axes_locatable(ax)

        # append a new axes on the right, same height as ax
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # draw colorbar in that axes
        cbar = fig.colorbar(potential_plot, cax=cax)

        # plot the sphere
        sphere = patches.Circle((0, 0), R0_init, edgecolor='black', fill=False, lw=3, zorder=5)
        ax.add_patch(sphere)

        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('z', fontsize=14)
        ax.grid()

    # Create the sliders
    R0_slider = FloatSlider(
        min=0.1*half_length, max=0.6*half_length, 
        step=slider_step, value=R0_init, 
        description='$R_{0}$'
        )
    rho_slider = FloatSlider(
        min=1, max=5, 
        step=slider_step, value=rho_init, 
        description='$\\rho$'
        )

    # Pack sliders in a box
    sliders = HBox([R0_slider, rho_slider])

    # update function
    def update(R0=0.3*half_length, rho=2):
        # update sphere
        sphere.set_radius(R0)
        # update potential values
        potential = gravitational_potential(R0, rho, R)
        # update potential plot
        potential_plot.set_array(potential.ravel())
        potential_plot.set_clim(vmin=0, vmax=rho*2*np.pi*(R0**2))
        cbar.update_normal(potential_plot)
        fig.canvas.draw_idle()

    # Update sliders
    widgets.interactive_output(update, {"R0": R0_slider, "rho": rho_slider})

    ui = HBox([out_fig, sliders])
    
    display(ui)


def sphere_gravity_potential_disturbed(half_length=20, slider_step=0.5, figure_size=7, reference=True):
    '''
    Plot a disturbed sphere and its gravitational potential in a 2D Cartesian system.
    The plot area extends from -"half_length" to +"half_length" 
    along x and y axes. The size of the vector (Cartesian) components 
    are controlled by sliders having step defined by "slider_step".
    The figure size is defined by "figure_size".
    '''

    assert isinstance(half_length, (float, int)), "half_length must be a scalar"
    assert half_length > 0, "half_length must be positive"
    assert isinstance(slider_step, (float, int)), "slider_step must be a scalar"
    assert slider_step > 0, "slider_step must be positive"
    assert isinstance(figure_size, (float, int)), "figure_size must be a scalar"
    assert figure_size > 0, "figure_size must be positive"

    area = [-half_length, half_length, -half_length, half_length]

    scale = 2 * half_length

    # Set computation points
    npoints = 100
    shape = (npoints, npoints)
    coordinates = np.linspace(-half_length, half_length, npoints)
    X = np.broadcast_to(coordinates, shape).T
    Z = np.broadcast_to(coordinates, shape)
    R = np.sqrt(X**2 + Z**2)

    # Set the initial radius
    R0_init = 0.3 * half_length
    # Set the initial density
    rho_init = 2

    # Set the initial anomalies
    R0a = 0.10 * R0_init
    rhoa = 500
    Xa_plus = 0.85 * R0_init * np.sqrt(2)/2
    Za_plus = 0.85 * R0_init * np.sqrt(2)/2
    Ra_plus = np.sqrt((X-Xa_plus)**2 + (Z-Za_plus)**2)
    Xa_minus = -0.85 * R0_init * np.sqrt(2)/2
    Za_minus = -0.85 * R0_init * np.sqrt(2)/2
    Ra_minus = np.sqrt((X-Xa_minus)**2 + (Z-Za_minus)**2)

    # create Output widget to hold the figure
    out_fig = widgets.Output()

    with out_fig:
        fig, ax = plt.subplots(figsize=(figure_size,figure_size))

        ax.set_aspect("equal")
        ax.set_xlim(area[0], area[1])
        ax.set_ylim(area[2], area[3])

        # compute the gravitational potential
        potential_plus = gravitational_potential(R0a, 0.01*rhoa*rho_init, Ra_plus)
        potential_minus = gravitational_potential(R0a, -0.01*rhoa*rho_init, Ra_minus)
        if reference is True:
            potential = gravitational_potential(R0_init, rho_init, R)
            potential_total = potential + potential_plus + potential_minus
            potential_max = rho_init*2*np.pi*(R0_init**2)
        else:
            potential_total = potential_plus + potential_minus
            potential_max = 0.01*abs(rhoa)*rho_init*2*np.pi*(R0a**2)

        # plot the gravitational potential
        potential_plot = ax.pcolormesh(
            X, Z, potential_total, 
            shading='nearest', cmap='seismic', 
            vmin=-potential_max, vmax=potential_max
            )

       # create a divider for the existing axes
        divider = make_axes_locatable(ax)

        # append a new axes on the right, same height as ax
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # draw colorbar in that axes
        cbar = fig.colorbar(potential_plot, cax=cax)

        # plot the sphere
        sphere = patches.Circle((0, 0), R0_init, edgecolor='black', fill=False, lw=3, zorder=5)
        sphere_plus = patches.Circle((Xa_plus, Za_plus), R0a, edgecolor='black', fill=False, lw=1, zorder=5)
        sphere_minus = patches.Circle((Xa_minus, Za_minus), R0a, edgecolor='black', fill=False, lw=1, zorder=5)
        ax.add_patch(sphere)
        ax.add_patch(sphere_plus)
        ax.add_patch(sphere_minus)

        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('z', fontsize=14)
        ax.grid()

    # Create the sliders
    R0_slider = FloatSlider(
        min=0.1*half_length, max=0.6*half_length, 
        step=slider_step, value=R0_init, 
        description='$R_{0}$'
        )
    rho_slider = FloatSlider(
        min=1, max=5, 
        step=slider_step, value=rho_init, 
        description='$\\rho$'
        )
    rhoa_slider = FloatSlider(
        min=-10000, max=10000, 
        step=100*slider_step, value=500, 
        description='$| \\rho_{a} |$'
        )

    # Pack sliders in a box
    sliders = HBox([R0_slider, rho_slider, rhoa_slider])

    # update function
    def update(R0=R0_init, rho=rho_init, rhoa=10):
        # Set the anomalies
        R0a = 0.10 * R0
        Xa_plus = 0.85 * R0 * np.sqrt(2)/2
        Za_plus = 0.85 * R0 * np.sqrt(2)/2
        Ra_plus = np.sqrt((X-Xa_plus)**2 + (Z-Za_plus)**2)
        Xa_minus = -0.85 * R0 * np.sqrt(2)/2
        Za_minus = -0.85 * R0 * np.sqrt(2)/2
        Ra_minus = np.sqrt((X-Xa_minus)**2 + (Z-Za_minus)**2)
        # update spheres
        sphere.set_radius(R0)
        sphere_plus.set_radius(R0a)
        sphere_plus.set_center((Xa_plus, Za_plus))
        sphere_minus.set_radius(R0a)
        sphere_minus.set_center((Xa_minus, Za_minus))
        # update potential values
        potential_plus = gravitational_potential(R0a, 0.01*rhoa*rho, Ra_plus)
        potential_minus = gravitational_potential(R0a, -0.01*rhoa*rho, Ra_minus)
        if reference is True:
            potential = gravitational_potential(R0, rho, R)
            potential_total = potential + potential_plus + potential_minus
            potential_max = rho*2*np.pi*(R0**2)
        else:
            potential_total = potential_plus + potential_minus
            potential_max = 0.01*abs(rhoa)*rho*2*np.pi*(R0a**2)
        # update potential plot
        potential_plot.set_array(potential_total.ravel())
        potential_plot.set_clim(vmin=-potential_max, vmax=potential_max)
        cbar.update_normal(potential_plot)
        fig.canvas.draw_idle()

    # Update sliders
    widgets.interactive_output(update, {"R0": R0_slider, "rho": rho_slider, "rhoa": rhoa_slider})

    ui = HBox([out_fig, sliders])
    
    display(ui)


# Function for computing the gravitational potential
def gravitational_potential(R0, rho, R):
    potential = np.zeros_like(R).ravel()
    mask = (R.ravel() > R0)
    potential[mask] = rho*(4/3)*np.pi*(R0**3)*(1/R.ravel()[mask])
    mask = (R.ravel() == R0)
    potential[mask] = rho*(4/3)*np.pi*(R0**2)
    mask = (R.ravel() < R0)
    potential[mask] = rho*2*np.pi*(R0**2 - (R.ravel()[mask]**2)/3)
    mask = (R.ravel() == 0)
    potential[mask] = rho*2*np.pi*(R0**2)
    return potential.reshape(R.shape)