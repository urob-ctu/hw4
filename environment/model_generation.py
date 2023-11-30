import numpy as np
from matplotlib.colors import hsv_to_rgb
from dm_control import mjcf
from dm_control import mujoco

def make_pendulum(id=0, hue=1, double=False):
    rgba = [0, 0, 0, 1]
    rgba[:3] = hsv_to_rgb([hue, 1, 1])
    rgba_transparent = [0, 0, 0, 0.6]
    rgba_transparent[:3] = rgba[:3]

    model = mjcf.RootElement(model='pendulum_%d' % id)

    model.default.geom.type = 'box'
    model.default.geom.rgba = rgba
    model.default.geom.conaffinity = 0
    model.default.geom.contype = 1
    model.default.geom.condim = 3
    model.default.geom.friction = (1, 0.5, 0.5)
    model.default.geom.margin = 0


    model.default.joint.damping = 0.15
    model.default.joint.armature = 0.0

    pend_len = 0.6
    pend_mass = 0.05
    cart_mass = 0.5
    cart_force = 100

    rail = model.worldbody.add('body', name='rail_body_%d' % id, pos=(0, 0, 0))
    rail.add('geom', name='rail_%d' % id, size=[0.02, 1], euler=[0, np.pi/2, 0], type='capsule')

    cart = rail.add('body', name='cart_%d' % id, pos=(0, 0, 0))
    slider = cart.add('joint', name='slider_joints_%d' % id, type='slide', axis=(1, 0, 0), limited=True, range=(-1, 1), damping=6)
    cart.add('geom', name='cart_geom_%d' % id, size=(0.1, 0.05, 0.05), mass=cart_mass)

    pole = cart.add('body', name='pole_%d' % id, pos=(0, 0, 0))
    pole.add('joint', type='hinge', axis=(0, 1, 0), damping=0.001)
    pole.add('geom', fromto=[0, 0, 0, 0.001, 0, pend_len], name="cpole", size=[0.02, 0.3], type="capsule", mass=pend_mass)

    if double:
        pole1 = pole.add('body', name='pole1_%d' % id, pos=(0, 0, pend_len))
        pole1.add('joint', type='hinge', axis=(0, 1, 0), damping=0.005)
        pole1.add('geom', fromto=[0, 0, 0, 0.001, 0, pend_len], name="cpole1", size=[0.02, 0.3], type="capsule", mass=1.2*pend_mass)

    model.actuator.add('motor', name='slider_%d' % id, joint=slider, gear=[cart_force, 0, 0, 0, 0, 0], ctrllimited=True, ctrlrange=(-1, 1))

    return model

def make_pendulum_sim(num_pends=1, frequency=100, double=False):
    arena = mjcf.RootElement(model='arena')

    arena.option.timestep = 1/frequency  # timestep for physics
    arena.option.density = 1.2  # enable density and viscosity of air to compute drag forces
    arena.option.viscosity = 0.00002
    arena.option.wind = (0, 0, 0)  # wind direction and speed
    # arena.option.integrator = 'RK4'

    arena.compiler.angle = 'radian'
    arena.compiler.inertiafromgeom = True
    arena.compiler.exactmeshinertia = True

    map_size = 15

    chequered = arena.asset.add('texture', name='checker', type='2d', builtin='checker', width=300,
                                height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
    grid = arena.asset.add('material', name='grid', texture=chequered,
                           texrepeat=[5, 5], reflectance=.2)
    arena.asset.add('texture', name='skybox', type='skybox', builtin='gradient',
                    rgb1=(.4, .6, .8), rgb2=(0, 0, 0), width=800, height=800, mark="random", markrgb=(1, 1, 1))
    arena.worldbody.add('geom', name='floor', type='plane', size=[map_size, map_size, 0.1], material=grid)

    arena.worldbody.add('light', directional='true', name='light', pos=[0, 0, 10], dir=[0, 0, -1.3])

    pendulums = [make_pendulum(i, i/num_pends, double) for i in range(num_pends)]
    height = 2
    sz = np.ceil(np.sqrt(num_pends)).astype(int)
    if sz < 8:
        y_steps = (np.arange(sz) - (sz-1)/2) * 0.5
        x_steps = (np.arange(sz) - (sz-1)/2) * 2.2
    else:
        xsz = np.ceil(sz/2).astype(int)
        ysz = sz*2
        y_steps = (np.arange(ysz) - (ysz-1)/2) * 0.5
        x_steps = (np.arange(xsz) - (xsz-1)/2) * 2.2
    xpos, ypos, zpos = np.meshgrid(x_steps, y_steps, [height])
    for i, model in enumerate(pendulums):
        spawn_pos = (xpos.flat[i], ypos.flat[i], zpos.flat[i])
        spawn_site = arena.worldbody.add('site', name='spawnsite_%d' % i, pos=spawn_pos, group=3)
        spawn_site.attach(model)
    return arena

def make_walker(id=0, hue=1):

    rgba = [0, 0, 0, 1]
    rgba[:3] = hsv_to_rgb([hue, 1, 1])
    rgba_transparent = [0, 0, 0, 0.6]
    rgba_transparent[:3] = rgba[:3]

    model = mjcf.RootElement(model='walker_%d' % id)

    model.default.geom.type = 'box'
    model.default.geom.rgba = rgba
    model.default.geom.conaffinity = 0
    model.default.geom.contype = 1
    model.default.geom.condim = 3
    model.default.geom.friction = (1, 0.5, 0.5)
    model.default.geom.margin = 0

    model.default.joint.damping = 2
    model.default.joint.armature = 0.15

    body = model.worldbody.add('body', name='walker_body%d' % id, pos=(0, 0, 0))
    body.add('geom', name='torso', type='capsule', size=[0.1, 0.1], mass=1.5)

    legs = ['left_leg', 'right_leg']
    positions = [[0, -0.13, -0.08], [0, 0.13, -0.08]]
    for i, leg in enumerate(legs):
        pos = positions[i]
        thigh = body.add('body', name='thigh_%d' % i, pos=pos)
        hip = thigh.add('joint', name='hip_%d' % i, type='hinge', axis=[0, 1, 0], range=[-2, 1.5], limited='true')
        thigh.add('geom', name='thigh_%d' % i, type='capsule', pos=[0, 0, -0.16], size=[0.05, 0.17], rgba=[0.3, 0.3, 0.3, 0.9], mass=0.4)
        shin = thigh.add('body', name='shin_%d' % i, pos=[0, 0, -0.3])
        knee = shin.add('joint', name='knee_%d' % i, type='hinge', axis=[0, 1, 0], range=[-0.2, 2.5], limited='true')
        shin.add('geom', name='shin_%d' % i, type='capsule', pos=[0, 0, -0.15], size=[0.045, 0.15], rgba=[0.3, 0.3, 0.3, 1], mass=0.2)
        shin.add('geom', name='foot_%d' % i, type='box', pos=[0.04, 0, -0.3], size=[0.09, 0.07, 0.04], rgba=[0.3, 0.3, 0.3, 1], mass=0.1)

        model.actuator.add('motor', name='%s_hip%d' % (leg, id), joint=hip, gear=[120, 0, 0, 0, 0, 0], ctrllimited=True, ctrlrange=(-1, 1))
        model.actuator.add('motor', name='%s_knee%d' % (leg, id), joint=knee, gear=[70, 0, 0, 0, 0, 0], ctrllimited=True, ctrlrange=(-1, 1))

    return model

def make_quadwalker(id=0, hue=1):

    rgba = [0, 0, 0, 1]
    rgba[:3] = hsv_to_rgb([hue, 1, 1])
    rgba_transparent = [0, 0, 0, 0.6]
    rgba_transparent[:3] = rgba[:3]

    model = mjcf.RootElement(model='walker_%d' % id)

    model.default.geom.type = 'box'
    model.default.geom.rgba = rgba
    model.default.geom.conaffinity = 0
    model.default.geom.contype = 1
    model.default.geom.condim = 3
    model.default.geom.friction = (1, 0.5, 0.5)
    model.default.geom.margin = 0

    model.default.joint.damping = 2
    model.default.joint.armature = 0.15

    body = model.worldbody.add('body', name='walker_body%d' % id, pos=(0, 0, 0))
    body.add('geom', name='torso', type='capsule', size=[0.1, 0.2], mass=1.5, euler=[0, np.pi/2, 0])

    legs = ['lf', 'rf', 'lb', 'rb']
    # make a spot-like dog robot
    positions = [[0.2, 0.13, 0], [0.2, -0.13, 0], [-0.2, 0.13, 0], [-0.2, -0.13,0]]
    for i, leg in enumerate(legs):
        pos = positions[i]
        x_off = 0.1 if i < 2 else -0.1
        y_off = 0.1*(-1)**i
        thigh = body.add('body', name='%s_thigh' % leg, pos=pos)
        hip = thigh.add('joint', name='%s_hip' % leg, type='hinge', axis=[0, 0, 1], range=[-0.5, 0.5], limited='true')
        thigh.add('geom', name='%s_thigh' % leg, type='capsule', fromto=[0,0,0,x_off,y_off,0], size=[0.05], rgba=[0.3, 0.3, 0.3, 1], mass=0.4)
        shin = thigh.add('body', name='shin_%d' % i, pos=[x_off, y_off, 0])

        knee_axis = [-y_off, x_off, 0]

        knee = shin.add('joint', name='knee_%d' % i, type='hinge', axis=knee_axis, range=[-1, 1], limited='true')
        shin.add('geom', name='shin_%d' % i, type='capsule', fromto=[0,0,0,0,0,-0.25], size=[0.03], rgba=[0.3, 0.3, 0.3, 1], mass=0.2)

        model.actuator.add('motor', name='%s_hip_motor%d' % (leg, id), joint=hip, gear=[60, 0, 0, 0, 0, 0], ctrllimited=True, ctrlrange=(-1, 1))
        model.actuator.add('motor', name='%s_knee_motor%d' % (leg, id), joint=knee, gear=[40, 0, 0, 0, 0, 0], ctrllimited=True, ctrlrange=(-1, 1))

    return model

def make_walker_sim(num_walkers=1, frequency=100):
    arena = mjcf.RootElement(model='arena')

    arena.option.timestep = 1/frequency  # timestep for physics
    arena.option.density = 1.2  # enable density and viscosity of air to compute drag forces
    arena.option.viscosity = 0.00002
    arena.option.wind = (0, 0, 0)  # wind direction and speed
    arena.option.integrator = 'RK4'

    arena.compiler.angle = 'radian'
    arena.compiler.inertiafromgeom = True
    arena.compiler.exactmeshinertia = True

    map_size = 15

    chequered = arena.asset.add('texture', name='checker', type='2d', builtin='checker', width=300,
                                height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
    grid = arena.asset.add('material', name='grid', texture=chequered,
                           texrepeat=[5, 5], reflectance=.2)
    arena.asset.add('texture', name='skybox', type='skybox', builtin='gradient',
                    rgb1=(.4, .6, .8), rgb2=(0, 0, 0), width=800, height=800, mark="random", markrgb=(1, 1, 1))
    arena.worldbody.add('geom', name='floor', type='plane', size=[map_size, map_size, 0.1], material=grid)

    arena.worldbody.add('light', directional='true', name='light', pos=[0, 0, 10], dir=[0, 0, -1.3])

    walkers = [make_quadwalker(i, i/num_walkers) for i in range(num_walkers)]
    height = 1
    sz = np.ceil(np.sqrt(num_walkers)).astype(int)
    dist = 1
    y_steps = (np.arange(sz) - (sz-1)/2) * dist
    x_steps = (np.arange(sz) - (sz-1)/2) * dist
    xpos, ypos, zpos = np.meshgrid(x_steps, y_steps, [height])
    for i, model in enumerate(walkers):
        spawn_pos = (xpos.flat[i], ypos.flat[i], zpos.flat[i])
        spawn_site = arena.worldbody.add('site', name='spawnsite_%d' % i, pos=spawn_pos, group=3)
        spawn_site.attach(model).add('freejoint')
    return arena

def mjcf_to_mjmodel(mjcf_model):
    xml_string = mjcf_model.to_xml_string(precision=5)
    # print(xml_string)
    # assets = arena.get_assets()
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model
