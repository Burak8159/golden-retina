"""Blender 4.5+ renderer for a parameterized, anatomy-informed synthetic eye-tracking set.

Run with: blender -b --python render_dataset.py -- --output ...
This creates non-identifiable 3D procedural faces; it is not a substitute for real-camera data.
"""
from __future__ import annotations
import bpy, math, random, csv, json, os, sys, struct, zlib, hashlib, time
from pathlib import Path
from mathutils import Vector
import numpy as np

W=512; H=512
CLASSES={"background":0,"face":1,"left_sclera":2,"left_iris":3,"left_pupil":4,"left_glint":5,
         "right_sclera":6,"right_iris":7,"right_pupil":8,"right_glint":9,"left_lid":10,"right_lid":11}

def set_input(node,names,value):
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value=value; return

def srgb(v):
    return v/12.92 if v<.04045 else ((v+.055)/1.055)**2.4

def mat(name,color,rough=.45,metal=0.0,subsurface=0.0,noise=False,transmission=0.0,emission=None):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    nt=m.node_tree;nt.nodes.clear();out=nt.nodes.new('ShaderNodeOutputMaterial');bs=nt.nodes.new('ShaderNodeBsdfPrincipled')
    set_input(bs,['Base Color'],(*color,1));set_input(bs,['Roughness'],rough);set_input(bs,['Metallic'],metal)
    set_input(bs,['Subsurface Weight','Subsurface'],subsurface);set_input(bs,['Transmission Weight','Transmission'],transmission)
    set_input(bs,['IOR'],1.376 if transmission else 1.45);set_input(bs,['Coat Weight','Clearcoat'],.28 if transmission else .04)
    if emission is not None:
        set_input(bs,['Emission Color','Emission'],(*emission,1));set_input(bs,['Emission Strength'],2.5)
    if noise:
        tex=nt.nodes.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=185;tex.inputs['Detail'].default_value=2.4;tex.inputs['Roughness'].default_value=.72
        coord=nt.nodes.new('ShaderNodeTexCoord');nt.links.new(coord.outputs['Generated'],tex.inputs['Vector'])
        bump=nt.nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.17;bump.inputs['Distance'].default_value=.012
        nt.links.new(tex.outputs['Fac'],bump.inputs['Height']);nt.links.new(bump.outputs['Normal'],bs.inputs['Normal'])
        ramp=nt.nodes.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.2;ramp.color_ramp.elements[1].position=.8
        base=(*color,1);dark=tuple(x*.90 for x in color)+(1,);light=tuple(min(1,x*1.07) for x in color)+(1,)
        ramp.color_ramp.elements[0].color=dark;ramp.color_ramp.elements[1].color=light
        nt.links.new(tex.outputs['Fac'],ramp.inputs['Fac']);nt.links.new(ramp.outputs['Color'],bs.inputs['Base Color'])
    nt.links.new(bs.outputs['BSDF'],out.inputs['Surface']);return m

def assign(obj,material,class_id):
    if material:obj.data.materials.append(material)
    obj.pass_index=int(class_id);return obj

def uv(name,loc,scale,material,class_id,segments=40,rings=28):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments,ring_count=rings,radius=1,location=loc)
    o=bpy.context.object;o.name=name;o.scale=scale
    for p in o.data.polygons:p.use_smooth=True
    return assign(o,material,class_id)

def curve(name,points,bevel,material,class_id,cyclic=False):
    c=bpy.data.curves.new(name,'CURVE');c.dimensions='3D';c.resolution_u=2;c.bevel_depth=bevel;c.bevel_resolution=2
    s=c.splines.new('POLY');s.points.add(len(points)-1)
    for p,co in zip(s.points,points):p.co=(*co,1)
    s.use_cyclic_u=cyclic;o=bpy.data.objects.new(name,c);bpy.context.collection.objects.link(o)
    return assign(o,material,class_id)

def mesh_obj(name,verts,faces,material,class_id):
    m=bpy.data.meshes.new(name);m.from_pydata(verts,[],faces);m.update();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o)
    for p in m.polygons:p.use_smooth=True
    return assign(o,material,class_id)

def age_sample(r):
    group=r.choices(['child','teen','adult','older_adult'],[.14,.14,.53,.19])[0]
    bounds={'child':(6,12),'teen':(13,19),'adult':(20,64),'older_adult':(65,85)}[group]
    return group,r.randint(*bounds)

def sample_params(seed):
    r=random.Random(seed);age_group,age=age_sample(r);presentation=r.choice(['feminine','masculine','androgynous'])
    tone=r.uniform(1,6);skin_pal=[(.97,.81,.70),(.84,.63,.48),(.67,.44,.31),(.48,.30,.22),(.32,.20,.15),(.19,.12,.095)]
    lo=int(math.floor(tone))-1;hi=min(5,lo+1);f=tone-math.floor(tone);skin=tuple(skin_pal[lo][i]*(1-f)+skin_pal[hi][i]*f for i in range(3))
    eye_color=r.choice([(0.20,.10,.045),(.20,.28,.20),(.12,.22,.34),(.32,.22,.10),(.18,.17,.16)])
    kid=age_group=='child';senior=age_group=='older_adult';fem=presentation=='feminine';masc=presentation=='masculine'
    return r,{'age_years':age,'age_group':age_group,'morphology_presentation':presentation,'skin_pigmentation_index':round(tone,2),
      'eye_color_linear':eye_color,'head_width':r.uniform(.75,.88)*(1.03 if kid else 1),
      'head_height':r.uniform(1.04,1.19)*(1.05 if kid else 1),'jaw_width':r.uniform(.69,.85)*(1.08 if masc else .94 if fem else 1),
      'brow_ridge':r.uniform(.025,.07)*(1.35 if masc else .82 if fem else 1),'cheek_fullness':r.uniform(.01,.08)*(1.35 if kid else 1),
      'ipd':r.uniform(.34,.43)*(1.02 if kid else 1),'eye_width':r.uniform(.135,.16),'eye_opening':r.uniform(.060,.082)*(1.08 if kid else 1),
      'iris_radius':r.uniform(.055,.070)*(1.04 if kid else 1),'pupil_radius':r.uniform(.020,.034),'gaze_x':r.uniform(-.065,.065),
      'gaze_z':r.uniform(-.045,.045),'lid_sag':r.uniform(.005,.038) if senior else r.uniform(0,.018),
      'lash_length':r.uniform(.008,.022)*(1.18 if fem else .88 if masc else 1),'blink_left':r.random()<.045,'blink_right':r.random()<.045,
      'head_yaw_deg':r.uniform(-16,16),'head_pitch_deg':r.uniform(-10,10),'head_roll_deg':r.uniform(-8,8),
      'camera_yaw_deg':r.uniform(-6,6),'camera_pitch_deg':r.uniform(-4,4),
      'capture_mode':r.choice(['visible','ir_like']), 'hair_style':r.choice(['short_cap','soft_cap','receded']) if not kid else r.choice(['soft_cap','short_cap']),
      'camera_distance':r.uniform(4.1,5.6),'camera_lens_mm':r.uniform(42,68),'blur_radius':r.choices([0,.25,.6],[.72,.2,.08])[0],
      'focus_offset_m':r.uniform(-.22,.22),
      'exposure':r.uniform(.72,1.28),'light_key_power':r.uniform(350,900),'light_fill_power':r.uniform(90,300),
      'eye_asymmetry':r.uniform(-.025,.025),'skin_color':skin,'skin_roughness':r.uniform(.31,.53),'seed':seed}

def create_head(p):
    skin=tuple(srgb(v) for v in p['skin_color']);skinmat=mat('Skin_microtexture',skin,p['skin_roughness'],subsurface=.065,noise=True)
    lidmat=mat('Eyelid_skin',tuple(v*.92 for v in skin),.39,subsurface=.07,noise=True)
    nostrilmat=mat('Nostril',(.035,.019,.016),.82);lipmat=mat('Lips',tuple(min(1,v*.82) for v in skin),.49,subsurface=.05)
    haircol=random.choice([(.012,.009,.008),(.035,.019,.012),(.075,.043,.024),(.16,.10,.055),(.28,.23,.18),(.42,.39,.34)] if p['age_group']=='older_adult' else [(.012,.009,.008),(.035,.019,.012),(.075,.043,.024),(.16,.10,.055),(.28,.23,.18)])
    hairmat=mat('Hair_strands',haircol,.78,noise=True);wrinklemat=mat('Fine_crease',tuple(v*.73 for v in skin),.57)
    # An ellipsoidal cranium is vertex-deformed into a chin/jaw/cheek silhouette.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=80,ring_count=64,radius=1,location=(0,0,0))
    head=bpy.context.object;head.name='Parametric_head_mesh';w=p['head_width'];h=p['head_height']
    for v in head.data.vertices:
        x,y,z=v.co;z0=z
        lower=max(0,min(1,(.22-z0)/1.22));x*=1-(1-p['jaw_width'])*lower
        cheek=sum(math.exp(-((x-s*.48)**2/.075+(z0+.08)**2/.16)) for s in (-1,1))*p['cheek_fullness']
        socket=sum(math.exp(-((x-s*p['ipd']/2)**2/.035+(z0-.31)**2/.018)) for s in (-1,1))
        v.co=(x*w,y*.66-cheek+socket*.025,z0*h)
    for f in head.data.polygons:f.use_smooth=True
    assign(head,skinmat,1)
    # Ears, nose bridge/tip, alae, nostrils, philtrum, lips and brow ridges.
    for s in (-1,1):
        uv('Auricle',(s*.77,-.04,.02),(.13,.105,.24),skinmat,1)
        uv('Ear_concha',(s*.825,-.119,.02),(.045,.02,.105),lidmat,1,24,16)
    uv('Nose_bridge',(0,-.605,.035),(.105,.155,.29),skinmat,1)
    uv('Nasal_tip',(0,-.727,-.115),(.125,.13,.092),skinmat,1)
    for s in (-1,1):
        uv('Nasal_ala',(s*.092,-.697,-.15),(.058,.067,.041),skinmat,1,28,18)
        uv('Nostril_opening',(s*.064,-.759,-.18),(.019,.009,.011),nostrilmat,1,20,12)
        uv('Brow_ridge',(s*.29,-.583,.49),(.255,.045,.043+p['brow_ridge']*.42),skinmat,1)
    uv('Upper_lip',(0,-.603,-.49),(.177,.035,.035),lipmat,1)
    uv('Lower_lip',(0,-.604,-.535),(.164,.038,.039),lipmat,1)
    uv('Chin',(0,-.49,-.79),(.245,.057,.11),skinmat,1)
    # Upper scalp shell only: a full ellipsoid here would sit in front of and
    # completely occlude the eyes. The open lower rim forms a controllable
    # hairline above the brows.
    if p['hair_style']!='receded':
        verts=[];faces=[];bands=20;around=64;center_z=.62
        for j in range(bands+1):
            theta=(math.pi*.5)*j/bands
            for k in range(around):
                phi=2*math.pi*k/around
                verts.append((.76*math.sin(theta)*math.cos(phi),.02+.64*math.sin(theta)*math.sin(phi),center_z+.58*math.cos(theta)))
        for j in range(bands):
            for k in range(around):
                a=j*around+k;b=j*around+(k+1)%around;c=(j+1)*around+(k+1)%around;d=(j+1)*around+k
                faces.append((a,b,c,d))
        cap=mesh_obj('Scalp_hair_cap',verts,faces,hairmat,1)
        for poly in cap.data.polygons:poly.use_smooth=True
        # Short strand curves follow the scalp surface to break the smooth
        # helmet silhouette. They are one curve object, not hundreds of objects.
        c=bpy.data.curves.new('Scalp_hair_fibers','CURVE');c.dimensions='3D';c.resolution_u=1;c.bevel_depth=.0011;c.bevel_resolution=1
        count=150 if p['hair_style']=='short_cap' else 95
        for _ in range(count):
            phi=random.uniform(-math.pi,math.pi);theta=random.uniform(.48,1.48);length=random.uniform(.045,.12)
            s=c.splines.new('POLY');s.points.add(3)
            for j,pt in enumerate(s.points):
                t=j/3;th=max(.06,theta-length*t);ph=phi+.025*math.sin(t*math.pi)
                nx=math.sin(th)*math.cos(ph);ny=math.sin(th)*math.sin(ph);nz=math.cos(th)
                pt.co=(.76*nx+.002*nx,.02+.64*ny+.002*ny,.62+.58*nz+.002*nz,1)
        hair=bpy.data.objects.new('Scalp_hair_fibers',c);bpy.context.collection.objects.link(hair);assign(hair,hairmat,1)
    for s in (-1,1):
        pts=[]
        for j in range(22):
            t=j/21;x=s*(.10+.47*t);z=.51+.025*math.sin(t*math.pi)-.045*t
            pts.append((x,-.663,z))
        curve('Eyebrow_hair',pts,.005,hairmat,1)
    if p['age_group']=='older_adult':
        for s in (-1,1):
            for k in range(2):
                pts=[(s*(.48+j*.025),-.665,.27-k*.045-j*.006) for j in range(7)]
                curve('Periorbital_crease',pts,.0025,wrinklemat,1)
    return skinmat,lidmat,hairmat

def make_lids(cx,cy,cz,half_w,opening,blink,sag,lash_len,skinmat,hairmat,lidclass):
    # Curved upper/lower lid bands in front of the spherical globe. Closed frames
    # have a real occluding lid surface, so eye labels disappear for the right reason.
    n=36;verts=[];faces=[];front=cy-.11;gap=0.007 if blink else opening*.90
    for j in range(n+1):
        t=-1+2*j/n;root=math.sqrt(max(0,1-t*t));x=cx+half_w*t
        aperture=gap*root
        upper_in=cz+aperture-sag*(.5+.5*t)
        upper_out=upper_in+.035+(.010*root)
        lower_in=cz-aperture-sag*(.20+.30*t)
        lower_out=lower_in-.031-.008*root
        y=front-.012*root
        verts.extend([(x,y,upper_in),(x,y+.012,upper_out),(x,y,lower_in),(x,y+.012,lower_out)])
    for j in range(n):
        a=j*4;b=(j+1)*4
        faces.extend([(a,b,b+1,a+1),(a+2,a+3,b+3,b+2)])
    mesh_obj('Curved_upper_lower_eyelid',verts,faces,skinmat,lidclass)
    # Tear-line margins and lashes lie on the opening edge.
    upper=[];lower=[]
    for j in range(n+1):
        t=-1+2*j/n;root=math.sqrt(max(0,1-t*t));x=cx+half_w*t;ap=gap*root;y=front-.02*root
        upper.append((x,y,cz+ap));lower.append((x,y,cz-ap))
    curve('Upper_eyelid_margin',upper,.0045,skinmat,lidclass)
    curve('Lower_tearline',lower,.0035,skinmat,lidclass)
    if blink:
        # A thin curved surface closes the palpebral aperture. A sphere made a
        # detached pillow and distorted the blink labels, so use a fitted patch.
        nx=48;ny=14;verts=[];faces=[]
        for i in range(nx+1):
            t=-1+2*i/nx;root=math.sqrt(max(0,1-t*t));x=cx+half_w*1.035*t
            for j in range(ny+1):
                # Extend slightly beyond the narrow residual aperture and place
                # the fitted surface toward the camera. The earlier patch left
                # a few pupil pixels visible on some head/gaze combinations.
                v=-1+2*j/ny;z=cz+opening*1.12*root*v;y=front-.028*root*(1-v*v)
                verts.append((x,y,z))
        for i in range(nx):
            for j in range(ny):
                a=i*(ny+1)+j;b=(i+1)*(ny+1)+j;faces.append((a,b,b+1,a+1))
        mesh_obj('Closed_eyelid_occluder',verts,faces,skinmat,lidclass)
        closed=[]
        for i in range(nx+1):
            t=-1+2*i/nx;root=math.sqrt(max(0,1-t*t));closed.append((cx+half_w*t,front-.031*root,cz+.004*math.sin(math.pi*t)))
        curve('Closed_lid_crease',closed,.002,skinmat,lidclass)
    else:
        count=7
        for j in range(count):
            t=-.84+1.68*j/(count-1);root=math.sqrt(max(0,1-t*t));x=cx+half_w*t
            x2=x+t*.018;z=cz+gap*root;y=front-.03*root
            curve('Lash_strand',[(x,y,z),(x2,y-.008,z+lash_len*(.68+.32*root))],.00125,hairmat,lidclass)

def make_eye(side,sign,params,skinmat,lidmat,hairmat):
    left=side=='left';eyeclass=2 if left else 6;irisclass=3 if left else 7;pupilclass=4 if left else 8;glintclass=5 if left else 9;lidclass=10 if left else 11
    cx=sign*params['ipd']/2;cz=.30+params['eye_asymmetry']*(1 if left else -1);cy=-.574
    halfw=params['eye_width'];opening=params['eye_opening'];blink=params['blink_left' if left else 'blink_right']
    eye=sphere_color=tuple(srgb(v) for v in params['eye_color_linear'])
    sclera_mat=mat(f'{side}_sclera',(srgb(.91),srgb(.89),srgb(.84)),.23,subsurface=.025)
    iris_mat=mat(f'{side}_iris',eye,.27,noise=True)
    pupil_mat=mat(f'{side}_pupil',(.002,.002,.002),.18)
    cornea_mat=mat(f'{side}_cornea',(.91,.94,.98),.025,transmission=.96)
    glint_mat=mat(f'{side}_glint',(1,1,1),.08,emission=(1,1,1))
    # Keep the globe naturally seated under the lid plane; a deep sphere made
    # the eyes read as protruding/cartoon-like in frontal renders.
    uv(f'{side}_eyeball_globe',(cx,cy,cz),(halfw,.096,opening*.82),sclera_mat,eyeclass,56,36)
    # Cornea/iris/pupil axis rotates from primary gaze; iris and pupil are nested
    # as actual spherical layers, with pupil dilation varied independently.
    gx=params['gaze_x']*(1 if left else .94);gz=params['gaze_z'];ix=cx+gx;iz=cz+gz;front=cy-.096
    ir=params['iris_radius'];pr=params['pupil_radius']
    uv(f'{side}_iris_disc',(ix,front-.006,iz),(ir,.012,ir),iris_mat,irisclass,48,32)
    # Limbus ring gives the iris a realistic dark anatomical boundary.
    bpy.ops.mesh.primitive_torus_add(major_radius=ir*.92,minor_radius=.006,major_segments=48,minor_segments=10,
      location=(ix,front-.009,iz),rotation=(math.pi/2,0,0))
    assign(bpy.context.object,mat(f'{side}_limbus',(eye[0]*.50,eye[1]*.47,eye[2]*.42),.3),irisclass)
    uv(f'{side}_pupil',(ix,front-.016,iz),(pr,.006,pr),pupil_mat,pupilclass,36,24)
    # A transmissive dome sits above the iris. It is omitted only during the ID
    # pass so labels describe the anatomical structures behind the cornea.
    cornea=uv(f'{side}_cornea_dome',(ix,front-.018,iz),(ir*1.12,.024,ir*1.12),cornea_mat,0,40,28)
    # Corneal glints follow the scene key-light direction and receive explicit labels.
    glx=ix-.032;glz=iz+.038
    glint=uv(f'{side}_corneal_glint',(glx,front-.042,glz),(.013,.006,.013),glint_mat,glintclass,20,12)
    glint_small=uv(f'{side}_corneal_glint_small',(ix+.045,front-.037,iz-.035),(.006,.003,.006),glint_mat,glintclass,16,10)
    if blink:
        # A closed lid covers the cornea and specular reflections as well as
        # the iris/pupil. Leaving frontal glints visible made some blinks read
        # as an open eye and produced impossible lid-over-glint labels.
        cornea.hide_render=True;glint.hide_render=True;glint_small.hide_render=True
    make_lids(cx,cy-.01,cz,halfw,opening,blink,params['lid_sag'],params['lash_length'],lidmat,hairmat,lidclass)
    return {'side':side,'center':(cx,cy,cz),'pupil_center':(ix,front-.016,iz),'iris_radius':ir,'pupil_radius':pr,'blink':blink,'corneas':[cornea]}

def add_lights_camera(params):
    r=random.Random(params['seed']+77)
    scene=bpy.context.scene;scene.world=bpy.data.worlds.new('studio_world') if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.18,.19,.22,1);scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.27
    for name,loc,power,size,color in [('key',(-2.8,-3.2,3.5),params['light_key_power'],2.0,(1.0,.91,.82)),('fill',(2.4,-2.5,1.3),params['light_fill_power'],2.8,(.78,.85,1.0)),('rim',(0,1.8,2.8),260,2.2,(1,.93,.85))]:
        bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.name=name;o.data.energy=power;o.data.shape='DISK';o.data.size=size;o.data.color=color
        o.rotation_euler=(Vector((0,-.4,.1))-o.location).to_track_quat('-Z','Y').to_euler()
    dist=params['camera_distance'];yaw=math.radians(params['camera_yaw_deg']);pitch=math.radians(params['camera_pitch_deg'])
    loc=(dist*math.sin(yaw),-dist*math.cos(yaw),.04+dist*math.tan(pitch))
    bpy.ops.object.camera_add(location=loc);cam=bpy.context.object;cam.name='capture_camera';cam.data.lens=params['camera_lens_mm'];cam.data.sensor_width=36
    cam.data.dof.use_dof=params['blur_radius']>0
    cam.data.dof.focus_distance=max(.2,dist+params['focus_offset_m']*params['blur_radius'])
    cam.data.dof.aperture_fstop=max(1.8,8.0-8.0*params['blur_radius'])
    cam.rotation_euler=(Vector((0,0,.02))-cam.location).to_track_quat('-Z','Y').to_euler();scene.camera=cam
    scene.render.resolution_x=W;scene.render.resolution_y=H;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
    scene.render.film_transparent=False;scene.view_settings.view_transform='AgX';scene.view_settings.exposure=math.log2(params['exposure'])
    scene.render.engine='CYCLES';scene.cycles.samples=24;scene.cycles.use_denoising=True;scene.cycles.device='GPU'
    scene.cycles.preview_samples=8
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    for d in prefs.devices:d.use=d.type=='OPTIX'
    scene.render.image_settings.compression=18
    scene.view_layers[0].use_pass_object_index=True
    return scene

def look_parameters(p):
    # Near-IR-like is a deliberately approximate material/camera style, not a
    # spectral simulation or a claim of matching a physical IR sensor.
    if p['capture_mode']=='ir_like':
        scale=random.uniform(.82,1.15)
        for m in bpy.data.materials:
            if m.use_nodes and 'Principled BSDF' in m.node_tree.nodes:
                bs=m.node_tree.nodes['Principled BSDF'];base=bs.inputs['Base Color']
                if not base.is_linked:
                    c=base.default_value;lum=.2126*c[0]+.7152*c[1]+.0722*c[2];g=max(.05,min(.92,lum*scale));base.default_value=(g,g,g,1)
                for ramp in (n for n in m.node_tree.nodes if n.bl_idname=='ShaderNodeValToRGB'):
                    for element in ramp.color_ramp.elements:
                        c=element.color;lum=.2126*c[0]+.7152*c[1]+.0722*c[2];g=max(.035,min(.92,lum*scale));element.color=(g,g,g,1)

def build_scene(seed,overrides=None):
    random.seed(seed);r,p=sample_params(seed)
    if overrides:
        unknown=set(overrides)-set(p)
        if unknown:raise ValueError(f'unsupported scene overrides: {sorted(unknown)}')
        p.update(overrides)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    # discard old generated datablocks to avoid memory growth across thousands of frames
    for datablocks in (bpy.data.meshes,bpy.data.curves,bpy.data.materials,bpy.data.cameras,bpy.data.lights):
        for d in list(datablocks):
            if d.users==0:datablocks.remove(d)
    skinmat,lidmat,hairmat=create_head(p);eyes=[make_eye('left',-1,p,skinmat,lidmat,hairmat),make_eye('right',1,p,skinmat,lidmat,hairmat)]
    # Rotate the complete face as one object hierarchy; vary camera pose separately.
    root=bpy.data.objects.new('head_pose_root',None);bpy.context.collection.objects.link(root)
    for o in [x for x in list(bpy.context.scene.objects) if x!=root]:
        mw=o.matrix_world.copy();o.parent=root;o.matrix_world=mw
    # Guard against an operator-context scale reset on the final right-eye
    # highlight sphere; the exact nominal size is part of the scene contract.
    for o in bpy.context.scene.objects:
        if o.name.endswith('corneal_glint_small'):
            o.scale=(.006,.003,.006)
    bpy.context.view_layer.update()
    for o in bpy.context.scene.objects:
        if o.name.endswith('corneal_glint_small'):
            if max(o.dimensions)>.03:raise RuntimeError(f'glint geometry scale invariant failed: {o.name} {tuple(o.dimensions)}')
    root.rotation_euler=(math.radians(p['head_pitch_deg']),math.radians(p['head_roll_deg']),math.radians(p['head_yaw_deg']))
    scene=add_lights_camera(p);look_parameters(p)
    bpy.context.view_layer.update()
    for e in eyes:
        for key in ('center','pupil_center'):
            e[key]=tuple(root.matrix_world @ Vector(e[key]))
    return scene,p,eyes

def write_png_gray(path,array):
    arr=np.asarray(array,dtype=np.uint8)
    if arr.shape!=(H,W):raise ValueError(f'bad mask shape {arr.shape}')
    raw=b''.join(b'\x00'+arr[y].tobytes() for y in range(H))
    def chunk(tag,data):return struct.pack('>I',len(data))+tag+data+struct.pack('>I',zlib.crc32(tag+data)&0xffffffff)
    payload=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',W,H,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b'')
    Path(path).write_bytes(payload)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def get_index_mask(scene,corneas,index_path):
    for o in corneas:o.hide_render=True
    old=(scene.render.filepath,scene.render.image_settings.file_format,scene.render.image_settings.color_mode,
         scene.render.image_settings.color_depth,scene.cycles.samples,scene.cycles.use_denoising,scene.use_nodes)
    try:
        scene.cycles.samples=1;scene.cycles.use_denoising=False
        # Write the raw IndexOB pass as float OpenEXR. This avoids depending on
        # Render Result internals, which vary across Blender builds.
        scene.use_nodes=True;tree=scene.node_tree;tree.nodes.clear()
        rl=tree.nodes.new('CompositorNodeRLayers');comp=tree.nodes.new('CompositorNodeComposite')
        if 'IndexOB' not in rl.outputs:raise RuntimeError('Compositor has no IndexOB output')
        tree.links.new(rl.outputs['IndexOB'],comp.inputs['Image'])
        scene.render.filepath=str(index_path)
        scene.render.image_settings.file_format='OPEN_EXR';scene.render.image_settings.color_mode='BW';scene.render.image_settings.color_depth='32'
        bpy.ops.render.render(write_still=True)
        actual=Path(str(index_path)+'.exr')
        if not actual.exists():
            candidates=sorted(index_path.parent.glob(index_path.name+'*.exr'))
            if not candidates:raise RuntimeError('Blender did not write the IndexOB EXR')
            actual=candidates[0]
        image=bpy.data.images.load(str(actual),check_existing=False)
        channels=int(image.channels);pix=np.asarray(image.pixels[:],dtype=np.float32)
        if channels==1 and pix.size==W*H:idx=pix.reshape(H,W)
        elif pix.size==W*H*channels:idx=pix.reshape(H,W,channels)[...,0]
        else:raise RuntimeError(f'unexpected IndexOB EXR dimensions/channels: {image.size[:]} / {channels} / {pix.size}')
        if not np.isfinite(idx).all():raise RuntimeError('IndexOB EXR contains non-finite values')
        if os.environ.get('GOLDEN_RETINA_DEBUG_INDEX')=='1':
            vals,counts=np.unique(np.round(idx,4),return_counts=True);print('DEBUG_INDEX',list(zip(vals[:30].tolist(),counts[:30].tolist())),flush=True)
        mask=np.rint(np.clip(idx,0,11)).astype(np.uint8)[::-1].copy()
        bpy.data.images.remove(image)
        actual.unlink(missing_ok=True)
        return mask
    finally:
        scene.render.filepath,scene.render.image_settings.file_format,scene.render.image_settings.color_mode,scene.render.image_settings.color_depth,scene.cycles.samples,scene.cycles.use_denoising,scene.use_nodes=old
        for o in corneas:o.hide_render=False

def project_point(scene,point):
    from bpy_extras.object_utils import world_to_camera_view
    q=world_to_camera_view(scene,scene.camera,Vector(point));return [round(q.x,6),round(1-q.y,6),round(q.z,6)]

def main():
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--train',type=int,default=1200);ap.add_argument('--validation',type=int,default=200);ap.add_argument('--test',type=int,default=300);ap.add_argument('--seed',type=int,default=815900);ap.add_argument('--samples',type=int,default=24);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args(args)
    out=Path(a.output);(out/'images').mkdir(parents=True,exist_ok=True);(out/'masks').mkdir(parents=True,exist_ok=True)
    splits=[('train',a.train,0),('validation',a.validation,1_000_000),('test',a.test,2_000_000)]
    total=sum(n for _,n,_ in splits);total=min(total,a.limit) if a.limit else total;rows=[];started=time.time();done=0
    classes_file=out/'classes.json';classes_file.write_text(json.dumps(CLASSES,indent=2)+'\n')
    for split,count,offset in splits:
        for i in range(count):
            if a.limit and done>=a.limit:break
            seed=a.seed+offset+i;scene,p,eyes=build_scene(seed);scene.cycles.samples=a.samples
            stem=f'{split}_{seed:09d}';image_path=out/'images'/f'{stem}.png';mask_path=out/'masks'/f'{stem}.png'
            scene.render.filepath=str(image_path);bpy.ops.render.render(write_still=True)
            corneas=[o for e in eyes for o in e['corneas']]
            mask=get_index_mask(scene,corneas,out/'masks'/f'{stem}_index');write_png_gray(mask_path,mask)
            landmarks={}
            for e in eyes:
                landmarks[e['side']+'_pupil_center_uv']=project_point(scene,e['pupil_center'])
                landmarks[e['side']+'_eye_center_uv']=project_point(scene,e['center'])
            meta={k:v for k,v in p.items() if k not in ('skin_color','eye_color_linear')};meta['eye_color_linear']=list(p['eye_color_linear']);meta['skin_color_srgb']=list(p['skin_color']);meta['landmarks']=landmarks
            rows.append({'image':str(image_path),'mask':str(mask_path),'split':split,'subject_id':f'synth3d-{seed:09d}','seed':seed,
                         'capture_mode':p['capture_mode'],'age_group':p['age_group'],'age_years':p['age_years'],'morphology_presentation':p['morphology_presentation'],
                         'skin_pigmentation_index':p['skin_pigmentation_index'],'left_blink':int(p['blink_left']),'right_blink':int(p['blink_right']),
                         'conditions_json':json.dumps(meta,separators=(',',':'))})
            done+=1
            if done%20==0 or done==total:
                print(json.dumps({'rendered':done,'requested':total,'split':split,'elapsed_min':round((time.time()-started)/60,2),'rate_per_min':round(done/max((time.time()-started)/60,.001),1)}),flush=True)
    manifest=out/'manifest.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    config={'renderer':'Blender 4.5.11 LTS Cycles OptiX','resolution_wh':[W,H],'samples':a.samples,'classes':CLASSES,
            'split_seed_offsets':{s:off for s,_,off in splits},'counts_requested':{s:n for s,n,_ in splits},
            'note':'Procedural 3D synthetic non-identifiable faces. IR-like is a material approximation, not a spectral camera model.'}
    (out/'renderer_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'status':'RENDER_GENERATION_COMPLETE','generated':done,'manifest_sha256':sha(manifest),'elapsed_min':round((time.time()-started)/60,2),'output':str(out)},indent=2),flush=True)
if __name__=='__main__':main()
