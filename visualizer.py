"""
3D VPython visualization of the tumbling rigid body, modeled as a detailed
wingnut -- the classic object used to demonstrate the Dzhanibekov Effect
(a hex nut threaded onto a bolt, with two tapered "wing" ears for
finger-tightening).

Environment features beyond the wingnut itself:
  - translucent extension lines continuing past each body-axis arrow tip
  - trailing "ray" trails at each axis arrow's tip, so the path each axis
    traces through its tumble can be followed visually over time
  - a translucent grid room (floor/ceiling/4 walls) with a marked start
    point, instead of a flat black backdrop, so the wingnut's motion reads
    clearly against a fixed frame of reference
  - a slow, purely illustrative constant linear drift (NOT derived from the
    physics engine, which only integrates orientation/angular velocity),
    with a translucent grey line tracking the straight-line path traveled
    so the object's own motion through space is visible, not just its spin

Design note on concurrency: we deliberately avoid the `threading` module
(not in the allowed library list) by NOT calling vpython's blocking rate()
loop from inside the Tkinter app. Instead, the Tkinter GUI drives timing via
`root.after(...)`, and on every tick it just re-assigns a handful of vpython
object attributes (wingnut.axis, wingnut.up, arrow.axis, ...). VPython's own
internal server thread (started by the vpython package itself, not by our
code) takes care of pushing those attribute changes to the browser tab
asynchronously. This keeps the two "loops" (Tkinter's and the browser's)
decoupled while still only importing vpython/tkinter/numpy/time/sys from
our own code.
"""

import numpy as np
import vpython as vp

from physics import quat_to_rotmat


# ---- wingnut geometry (all built along LOCAL X, the compound's natural
# "axis" direction, so update_scene can simply reassign .axis/.up) -----

NUT_HALF_LENGTH = 0.35      # hex nut collar half-length along the shaft
NUT_RADIUS = 0.45           # hex nut "radius" (flat-to-flat-ish)
SHAFT_RADIUS = 0.09         # smooth bolt core radius
SHAFT_HALF_LENGTH = 1.3     # bolt extends this far either side of center
THREAD_RADIUS = 0.115       # thread ridge radius, just outside the shaft
THREAD_RING_THICKNESS = 0.018
N_THREAD_RINGS_PER_SIDE = 8

WING_THICKNESS = 0.13       # z-extent (flat ear thickness)
WING_SEGMENTS = [           # (x_width, y_center, y_half_span) tapering out
    (0.55, 0.28, 0.28),
    (0.40, 0.62, 0.20),
    (0.24, 0.86, 0.12),
]
WING_TIP_RADIUS = 0.075     # small rounded grip knob at each wingtip

NUT_COLOR = vp.vector(0.75, 0.76, 0.78)     # brushed steel
SHAFT_COLOR = vp.vector(0.82, 0.83, 0.85)   # lighter steel
THREAD_COLOR = vp.vector(0.45, 0.46, 0.48)  # darker groove shading
WING_COLOR = vp.vector(0.95, 0.55, 0.10)    # high-visibility orange

#axis-arrow translucent extensions

AXIS_ARROW_LEN = 1.8
AXIS_EXT_LEN = 3.2          # extension length beyond each arrow's tip
AXIS_EXT_RADIUS = 0.02
AXIS_EXT_OPACITY = 0.28

#background: translucent grid room + start-point marker

ROOM_HALF = 6.0             # room spans -6..6 on each axis (12-unit cube)
GRID_SPACING = 1.0
GRID_MAJOR_EVERY = 3        # every 3rd line is drawn brighter/thicker
GRID_MINOR_COLOR = vp.vector(0.32, 0.34, 0.38)
GRID_MAJOR_COLOR = vp.vector(0.55, 0.58, 0.64)
GRID_LINE_RADIUS_MINOR = 0.008
GRID_LINE_RADIUS_MAJOR = 0.016
WALL_COLOR = vp.vector(0.45, 0.5, 0.6)
WALL_OPACITY = 0.06

START_MARKER_RADIUS = 0.5
START_MARKER_COLOR = vp.color.yellow

#axis-tip trailing "rays" (native vpython trails)

TIP_MARKER_RADIUS = 0.035
TRAIL_RETAIN = 140
TRAIL_RADIUS = 0.012

#linear drift (purely a visual overlay -- see module docstring) 

DRIFT_VELOCITY = np.array([0.045, 0.018, 0.028])   # units/second
MOTION_LINE_RADIUS = 0.02
MOTION_LINE_COLOR = vp.color.gray(0.6)
MOTION_LINE_OPACITY = 0.35


def _grid_plane_lines(canvas, fixed_axis, fixed_value, extent, spacing, major_every):
    """Draw a two-family grid of curve lines on the plane where coordinate
    `fixed_axis` == fixed_value, spanning +/- extent along the other two
    axes. Returns the list of curve objects."""
    other = [a for a in (0, 1, 2) if a != fixed_axis]
    n = int(round(extent / spacing))
    lines = []
    for i in range(-n, n + 1):
        coord = i * spacing
        is_major = (i % major_every == 0)
        color = GRID_MAJOR_COLOR if is_major else GRID_MINOR_COLOR
        radius = GRID_LINE_RADIUS_MAJOR if is_major else GRID_LINE_RADIUS_MINOR

        # line varying along other[0], held at other[1] = coord
        p1, p2 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        p1[fixed_axis] = p2[fixed_axis] = fixed_value
        p1[other[1]] = p2[other[1]] = coord
        p1[other[0]], p2[other[0]] = -extent, extent
        lines.append(vp.curve(canvas=canvas, pos=[vp.vector(*p1), vp.vector(*p2)],
                               color=color, radius=radius))

        # line varying along other[1], held at other[0] = coord
        p1, p2 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        p1[fixed_axis] = p2[fixed_axis] = fixed_value
        p1[other[0]] = p2[other[0]] = coord
        p1[other[1]], p2[other[1]] = -extent, extent
        lines.append(vp.curve(canvas=canvas, pos=[vp.vector(*p1), vp.vector(*p2)],
                               color=color, radius=radius))
    return lines


def _build_room(canvas):
    """A translucent 12-unit cube 'room' (floor, ceiling, 4 walls) with a
    grid etched into each face. Unlike a starfield, a grid gives continuous
    reference lines the eye can track the wingnut moving past/through as it
    drifts and tumbles -- much stronger motion cues than empty black."""
    faces = [
        (1, -ROOM_HALF),  # floor
        (1, ROOM_HALF),   # ceiling
        (0, -ROOM_HALF),  # wall -x
        (0, ROOM_HALF),   # wall +x
        (2, -ROOM_HALF),  # wall -z
        (2, ROOM_HALF),   # wall +z
    ]
    panels, all_lines = [], []
    for fixed_axis, fixed_value in faces:
        size = [2 * ROOM_HALF] * 3
        size[fixed_axis] = 0.05
        pos = [0.0, 0.0, 0.0]
        pos[fixed_axis] = fixed_value
        panel = vp.box(canvas=canvas, pos=vp.vector(*pos), size=vp.vector(*size),
                        color=WALL_COLOR, opacity=WALL_OPACITY, shininess=0.05)
        panels.append(panel)
        all_lines.extend(_grid_plane_lines(canvas, fixed_axis, fixed_value,
                                            ROOM_HALF, GRID_SPACING, GRID_MAJOR_EVERY))
    return {"panels": panels, "lines": all_lines}


def _build_start_marker(canvas):
    """A crosshair of rings at the world origin -- where the wingnut's drift
    begins -- plus a floor marker and a faint vertical guide line, so the
    grid room has a clear 'markings showing the start point'."""
    rings = [
        vp.ring(canvas=canvas, pos=vp.vector(0, 0, 0), axis=vp.vector(1, 0, 0),
                radius=START_MARKER_RADIUS, thickness=0.02, color=START_MARKER_COLOR),
        vp.ring(canvas=canvas, pos=vp.vector(0, 0, 0), axis=vp.vector(0, 1, 0),
                radius=START_MARKER_RADIUS, thickness=0.02, color=START_MARKER_COLOR),
        vp.ring(canvas=canvas, pos=vp.vector(0, 0, 0), axis=vp.vector(0, 0, 1),
                radius=START_MARKER_RADIUS, thickness=0.02, color=START_MARKER_COLOR),
    ]
    label = vp.label(canvas=canvas, pos=vp.vector(0, -0.95, 0), text="START",
                      box=False, height=12, color=START_MARKER_COLOR)
    floor_mark = vp.ring(canvas=canvas, pos=vp.vector(0, -ROOM_HALF + 0.02, 0),
                          axis=vp.vector(0, 1, 0), radius=0.4, thickness=0.03,
                          color=START_MARKER_COLOR)
    guide = vp.cylinder(canvas=canvas, pos=vp.vector(0, -ROOM_HALF, 0),
                         axis=vp.vector(0, ROOM_HALF, 0), radius=0.008,
                         color=START_MARKER_COLOR, opacity=0.25)
    return {"rings": rings, "label": label, "floor_mark": floor_mark, "guide": guide}


def _build_wingnut(canvas):
    """Assemble the hex nut + threaded shaft + tapered wings as a single
    rigid vp.compound, built along local X so its native .axis/.up align
    with our rotation-matrix columns (ex, ey)."""
    parts = []

    # smooth bolt core running the full length
    shaft = vp.cylinder(
        canvas=canvas, pos=vp.vector(-SHAFT_HALF_LENGTH, 0, 0),
        axis=vp.vector(2 * SHAFT_HALF_LENGTH, 0, 0),
        radius=SHAFT_RADIUS, color=SHAFT_COLOR, shininess=0.6,
    )
    parts.append(shaft)

    # threaded ridges exposed on either side of the nut, represented as a
    # stack of thin rings (vp.helix cannot be used inside a vp.compound,
    # so rings are the compound-compatible way to suggest thread grooves)
    exposed_len = SHAFT_HALF_LENGTH - NUT_HALF_LENGTH
    for sign in (+1, -1):
        start_x = sign * NUT_HALF_LENGTH
        for i in range(N_THREAD_RINGS_PER_SIDE):
            frac = (i + 0.5) / N_THREAD_RINGS_PER_SIDE
            x = start_x + sign * frac * exposed_len
            ridge = vp.ring(
                canvas=canvas, pos=vp.vector(x, 0, 0), axis=vp.vector(1, 0, 0),
                radius=THREAD_RADIUS, thickness=THREAD_RING_THICKNESS,
                color=THREAD_COLOR,
            )
            parts.append(ridge)

    # hexagonal nut collar, extruded along the shaft axis
    nut = vp.extrusion(
        canvas=canvas,
        path=[vp.vector(-NUT_HALF_LENGTH, 0, 0), vp.vector(NUT_HALF_LENGTH, 0, 0)],
        shape=vp.shapes.hexagon(length=NUT_RADIUS),
        color=NUT_COLOR, shininess=0.5,
    )
    parts.append(nut)

    # two tapered "bowtie" wing ears (+y and -y), each built from shrinking
    # box segments plus a small rounded tip knob for grip
    for sign in (+1, -1):
        for (x_width, y_center, y_half_span) in WING_SEGMENTS:
            seg = vp.box(
                canvas=canvas,
                pos=vp.vector(0, sign * y_center, 0),
                size=vp.vector(x_width, 2 * y_half_span, WING_THICKNESS),
                color=WING_COLOR, shininess=0.3,
            )
            parts.append(seg)
        tip_y = sign * (WING_SEGMENTS[-1][1] + WING_SEGMENTS[-1][2])
        tip = vp.sphere(
            canvas=canvas, pos=vp.vector(0, tip_y, 0),
            radius=WING_TIP_RADIUS, color=WING_COLOR, shininess=0.3,
        )
        parts.append(tip)

    wingnut = vp.compound(parts, canvas=canvas)
    return wingnut


def init_scene(title="Dzhanibekov Effect - Wingnut Tumble"):
    """
    Create the VPython canvas with the detailed wingnut model, a translucent
    grid room with a marked start point, body-fixed axis arrows with
    translucent extensions and trailing tip "rays", an angular-velocity
    arrow, and a translucent line tracking the wingnut's (illustrative)
    linear drift through space.
    Returns a dict of handles used by update_scene().
    """
    scene = vp.canvas(title=title, width=900, height=560,
                       background=vp.color.gray(0.06), align="left")
    scene.forward = vp.vector(-1, -0.6, -1)
    scene.range = 6.5
    scene.lights = []
    vp.distant_light(canvas=scene, direction=vp.vector(0.4, 0.6, 1), color=vp.color.gray(0.9))
    vp.distant_light(canvas=scene, direction=vp.vector(-0.6, -0.3, -0.8), color=vp.color.gray(0.4))

    # background detail -- a grid room with a marked start point gives the
    # eye continuous reference lines to track the wingnut's motion against,
    # instead of a flat black backdrop that hides movement entirely
    room = _build_room(scene)
    start_marker = _build_start_marker(scene)

    wingnut = _build_wingnut(scene)
    axis_length = wingnut.axis.mag  # capture native length before we rotate it
    scene.camera.follow(wingnut)    # keep the drifting object framed

    x_arrow = vp.arrow(canvas=scene, pos=vp.vector(0, 0, 0), axis=vp.vector(AXIS_ARROW_LEN, 0, 0),
                        color=vp.color.red, shaftwidth=0.05)
    y_arrow = vp.arrow(canvas=scene, pos=vp.vector(0, 0, 0), axis=vp.vector(0, AXIS_ARROW_LEN, 0),
                        color=vp.color.green, shaftwidth=0.05)
    z_arrow = vp.arrow(canvas=scene, pos=vp.vector(0, 0, 0), axis=vp.vector(0, 0, AXIS_ARROW_LEN),
                        color=vp.color.blue, shaftwidth=0.05)

    # translucent lines continuing past each arrow's tip, emphasizing that
    # these body axes extend as reference directions beyond the arrowheads
    x_ext = vp.cylinder(canvas=scene, pos=vp.vector(AXIS_ARROW_LEN, 0, 0),
                         axis=vp.vector(AXIS_EXT_LEN, 0, 0), radius=AXIS_EXT_RADIUS,
                         color=vp.color.red, opacity=AXIS_EXT_OPACITY)
    y_ext = vp.cylinder(canvas=scene, pos=vp.vector(0, AXIS_ARROW_LEN, 0),
                         axis=vp.vector(0, AXIS_EXT_LEN, 0), radius=AXIS_EXT_RADIUS,
                         color=vp.color.green, opacity=AXIS_EXT_OPACITY)
    z_ext = vp.cylinder(canvas=scene, pos=vp.vector(0, 0, AXIS_ARROW_LEN),
                         axis=vp.vector(0, 0, AXIS_EXT_LEN), radius=AXIS_EXT_RADIUS,
                         color=vp.color.blue, opacity=AXIS_EXT_OPACITY)

    # small markers riding at each arrow's tip, with a native VPython trail
    # enabled -- these leave a fading "ray" behind showing the recent path
    # that axis has traced through its tumble, so it can be followed by eye
    x_tip = vp.sphere(canvas=scene, pos=vp.vector(AXIS_ARROW_LEN, 0, 0),
                       radius=TIP_MARKER_RADIUS, color=vp.color.red,
                       make_trail=True, trail_color=vp.color.red,
                       retain=TRAIL_RETAIN, trail_radius=TRAIL_RADIUS)
    y_tip = vp.sphere(canvas=scene, pos=vp.vector(0, AXIS_ARROW_LEN, 0),
                       radius=TIP_MARKER_RADIUS, color=vp.color.green,
                       make_trail=True, trail_color=vp.color.green,
                       retain=TRAIL_RETAIN, trail_radius=TRAIL_RADIUS)
    z_tip = vp.sphere(canvas=scene, pos=vp.vector(0, 0, AXIS_ARROW_LEN),
                       radius=TIP_MARKER_RADIUS, color=vp.color.blue,
                       make_trail=True, trail_color=vp.color.blue,
                       retain=TRAIL_RETAIN, trail_radius=TRAIL_RADIUS)

    w_arrow = vp.arrow(canvas=scene, pos=vp.vector(0, 0, 0), axis=vp.vector(0, 0, 0),
                        color=vp.color.yellow, shaftwidth=0.06)

    # translucent grey line from the drift's starting point to the wingnut's
    # current position -- grows every frame, making the linear motion
    # (on top of the tumbling) visually explicit
    motion_line = vp.cylinder(canvas=scene, pos=vp.vector(0, 0, 0),
                               axis=vp.vector(0, 0, 0), radius=MOTION_LINE_RADIUS,
                               color=MOTION_LINE_COLOR, opacity=MOTION_LINE_OPACITY)

    return {
        "scene": scene, "body": wingnut, "axis_length": axis_length,
        "room": room, "start_marker": start_marker,
        "x_arrow": x_arrow, "y_arrow": y_arrow, "z_arrow": z_arrow,
        "x_ext": x_ext, "y_ext": y_ext, "z_ext": z_ext,
        "x_tip": x_tip, "y_tip": y_tip, "z_tip": z_tip,
        "w_arrow": w_arrow, "motion_line": motion_line,
    }


def update_scene(handles, q, w, t=0.0, axis_len=AXIS_ARROW_LEN, w_scale=0.35):
    """
    Update the wingnut orientation/position and all the frame decorations
    given the current quaternion `q` (w,x,y,z; body->inertial), body angular
    velocity `w`, and elapsed simulation time `t` (used only to drive the
    illustrative linear drift -- rotation itself doesn't depend on t here
    since q already encodes the integrated orientation at this instant).
    """
    R = quat_to_rotmat(q)  # columns = body x,y,z expressed in inertial frame
    ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]

    body_pos = vp.vector(*(DRIFT_VELOCITY * t))

    body = handles["body"]
    body.pos = body_pos
    body.axis = vp.vector(*(ex * handles["axis_length"]))
    body.up = vp.vector(*ey)

    # body-fixed axis arrows + their translucent extensions, all translated
    # along with the wingnut so they represent its axes at its current spot
    ex_vec = vp.vector(*(ex * axis_len))
    ey_vec = vp.vector(*(ey * axis_len))
    ez_vec = vp.vector(*(ez * axis_len))

    handles["x_arrow"].pos = body_pos
    handles["x_arrow"].axis = ex_vec
    handles["y_arrow"].pos = body_pos
    handles["y_arrow"].axis = ey_vec
    handles["z_arrow"].pos = body_pos
    handles["z_arrow"].axis = ez_vec

    handles["x_ext"].pos = body_pos + ex_vec
    handles["x_ext"].axis = vp.vector(*(ex * AXIS_EXT_LEN))
    handles["y_ext"].pos = body_pos + ey_vec
    handles["y_ext"].axis = vp.vector(*(ey * AXIS_EXT_LEN))
    handles["z_ext"].pos = body_pos + ez_vec
    handles["z_ext"].axis = vp.vector(*(ez * AXIS_EXT_LEN))

    # move each tip marker to the arrow's current tip; VPython's native
    # make_trail mechanism automatically extends a fading "ray" behind it
    handles["x_tip"].pos = body_pos + ex_vec
    handles["y_tip"].pos = body_pos + ey_vec
    handles["z_tip"].pos = body_pos + ez_vec

    w_inertial = R @ np.asarray(w)
    handles["w_arrow"].pos = body_pos
    handles["w_arrow"].axis = vp.vector(*(w_inertial * w_scale))

    # grey translucent line from the drift's start to the current position
    handles["motion_line"].axis = body_pos


if __name__ == "__main__":
    # standalone demo: run a simulation and play it back directly (blocking),
    # useful for testing visualizer.py without the Tkinter GUI.
    from physics import RigidBody

    body_phys = RigidBody(I1=1.0, I2=2.0, I3=3.0)
    result = body_phys.simulate(w0=[0.05, 5.0, 0.05], t_span=(0, 20), n_points=1500)

    handles = init_scene()
    t = result["t"]
    q_arr = result["q"]
    w_arr = result["w"]

    i = 0
    n = len(t)
    while i < n:
        vp.rate(60)
        update_scene(handles, q_arr[i], w_arr[i], t=t[i])
        i += 1
