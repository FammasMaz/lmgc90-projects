from pylmgc90 import pre
import math
import numpy as np

from pylmgc90 import chipy   
import timeit 

def rail_generator(nb_rails, rail_length, rail_height, rail_width, rail_spacing, rail_offset, rail_color, rail_material, rail_model):
    bodies = pre.avatars()
    for i in range(nb_rails):
        rail = pre.buildMeshH8(x0=-rail_length/2., y0=-rail_height/2., z0=0., lx=rail_length, ly=rail_height, lz=rail_width, nb_elem_x=2, nb_elem_y=2, nb_elem_z=2)
        rail = pre.buildMeshedAvatar(mesh=rail, model=rail_model, material=rail_material)
        rail.addContactors(group='down', shape='CSpxx', color=rail_color)
        rail.translate(dz=rail_offset , dy=rail_height*1.5-rail_spacing*i)
        rail.rotate(description='axis',alpha=math.pi/2.,axis=[1.,0.,0.],center=rail.nodes[1].coor)
        bodies.addAvatar(rail)
    return bodies

def ballast_generator(nb_particles, Rmin, Rmax, Px, Py, Pz, layers, mat, mod, ptype, min_vert, max_vert, gen_type='box', seed=43, color='BLUEx', part_dist=None, nb_layers=None):
   bodies = pre.avatars()
   total_particles = 0
   particle_char = {'body_id': [], 'radius': []}
   k=2
   for j in range(len(layers)):
      radii_x = pre.granulo_Uniform(np.int(nb_particles), Rmin, Rmax)
      #radii_x = pre.granulo_Uniform(np.int(nb_particles*layers[j]), Rmin, Rmax)
      if gen_type == 'box':
         [nb_rem,coors] = pre.depositInBox3D(radii=radii_x, lx=Px, ly=Py*layers[j], lz=Pz)
      else:
         nbx = int(Px/part_dist); nby = int(Py*layers[j]/part_dist)
         coors = pre.cubicLattice3D(nbx, nby, nb_layers, part_dist, x0 = -Px/2., y0 = -Py*layers[j]/2., z0 = 0.01)
         nb_rem = len(coors)//3
      for i in range(nb_rem):
          #if ptype == 'POLYR':
          body = pre.rigidPolyhedron(radius=radii_x[i], center=coors[3*i : 3*(i+1)], nb_vertices=np.random.randint(min_vert, max_vert), generation_type='random',model=mod,
                                material=mat, color=color)

              
          #else:
           #   body = pre.rigidSphere(r=radii_x[i],center=coors[3*i:3*(i+1)],model=mod,
            #                    material=mat,color=color)
          r=np.random.random(3)*np.pi
          #body.rotate(phi=r[0],theta=r[1],psi=r[2])
          body.translate(dz=Pz*(len(layers)-j) + Rmax*1.5)
          bodies.addAvatar(body)
          particle_char['body_id'].append(k)
          particle_char['radius'].append(radii_x[i])
          k+=1
      total_particles += nb_rem
   return bodies, total_particles, particle_char

def wall_generator(wall_length, wall_height, wall_width, wall_material, wall_model, rot_axis=None,alpha=math.pi/2., wall_offset=None, wall_color='WALLx'):
        bodies = pre.avatars()
        wall = pre.rigidPlan(axe1=wall_length/2.,axe2=wall_height/2.,axe3=wall_width/2.,center=[0.,0.,0.],
                       material=wall_material,model=wall_model,color=wall_color)
        [wall.translate(dx=wall_offset[0], dy=wall_offset[1], dz=wall_offset[2]) if wall_offset is not None else None]
        [wall.rotate(description='axis',alpha=alpha,axis=rot_axis,center=wall.nodes[1].coor) if rot_axis is not None else None]
        bodies.addAvatar(wall)
        return bodies

def trapezoid_generator(txb,txt, ty, tz, mat, mod, color='BLUEx'):
    body = pre.avatar(dimension=3)
    body.addBulk(pre.rigid3d())
   # create a trapezoid
  
    body.addNode(pre.node(np.array([0., -ty/2, 0.]), 1))
    body.addNode(pre.node(np.array([0., ty/2, 0.]), 1))
    body.addNode(pre.node(np.array([txb, -ty/2, 0.]), 1))
    body.addNode(pre.node(np.array([txb, ty/2, 0.]), 1))
    body.addNode(pre.node(np.array([txt, -ty/2, tz]), 1))
    body.addNode(pre.node(np.array([txt, ty/2, tz]), 1))
    body.addNode(pre.node(np.array([0., -ty/2, tz]), 1))
    body.addNode(pre.node(np.array([0., ty/2, tz]), 1))

    body.defineGroups()
    body.defineModel(model=mod)
    body.defineMaterial(material=mat)
    # define contactors on all the faces of the trapezoid
    #body.addContactors(shape='PLANx', color=color, group='all')
    body.computeRigidProperties()
    return body


    

def ballast_generator_custom(ballast_bib, layers, nb_particles, Px, Py, Pz, mat, mod, Rmin=0.3, Rmax=0.4, ):
  bodies = pre.avatars()
  total_particles = 0
  particle_char = {'body_id': [], 'radius': []}
  l=2
  lballast = get_lbody(ballast_bib)
  for j in range(len(layers)):
    radii_x = pre.granulo_Random(int(nb_particles), Rmin, Rmax)
    [nb_dep, coors] = pre.depositInBox3D(radii_x, Px, Py*layers[j], Pz)
    x = coors[::3];  y = coors[1::3];  z = coors[2::3]
    for k in range(nb_dep):
      rand = int(np.random.random()*len(lballast))
      body = pre.rigidPolyhedron(mod, mat, color='BLUEx', generation_type='full', vertices=lballast[rand][0], faces=lballast[rand][1])
      r = np.random.random(3)*np.pi
      body.rotate(phi=r[0],theta=r[1],psi=r[2])
      body.translate(dx=x[k], dy=y[k], dz=z[k])
      # translate up 
      if len(layers)==1: body.translate(dz=Rmax*3)
      else:body.translate(dz=Pz*(len(layers)-j) + Rmax*3)

      bodies.addAvatar(body)
      particle_char['body_id'].append(l)
      particle_char['radius'].append(radii_x[k])
      l+=1
    total_particles += nb_dep
  return bodies, total_particles, particle_char

def ballast_generator_filled(ballast_bib, lengthb, widthb, heightb, mat, mod, nb_particles = 5206):
   bodies = pre.avatars()
   lballast = get_lbody(ballast_bib)
   radii = pre.granulo_Random(int(nb_particles), 4.4044198827808083E-002, 5.7470355839992146E-002)
   elapsed = 0
   threshold_time = 0.5
   while elapsed < threshold_time:
         start_time = timeit.default_timer()
         radii = pre.granulo_Random(int(nb_particles), 2.4044198827808083E-002, 5.7470355839992146E-002)
         [nb_dep, coor] = pre.depositInBox3D(radii, lengthb, widthb, heightb)
         elapsed = timeit.default_timer() - start_time
   print(f'Elapsed time for depositInBox3D: {elapsed}')
   x = coor[::3];  y = coor[1::3];  z = coor[2::3]
   i=0
   for i in range(nb_dep):
        rand = int(np.random.random()*len(lballast))
        body = pre.rigidPolyhedron(mod, mat, color='BLUEx', generation_type='full', vertices=lballast[rand][0], faces=lballast[rand][1])
        r = np.random.random(3)*np.pi
        body.rotate(phi=r[0],theta=r[1],psi=r[2])
        body.translate(dx=x[i], dy=y[i], dz=z[i])
        bodies.addAvatar(body)
        i+=1

   return bodies, nb_dep,z


def ballast_generator_closet(ballast_bib, nbx, nby, nlayer, dgrid, lengthb, widthb, mat, model, color='BLUEx', angle_for_plate=0.):
    bodies = pre.avatars()
    lballast = get_lbody(ballast_bib)
    coor = pre.cubicLattice3D(nbx, nby, nlayer, dgrid, x0=-lengthb/2, y0=-widthb/2, z0=0.01)
    x = coor[0::3]
    y = coor[1::3]  
    z = coor[2::3]
    i=0
    Ngrains = nbx*nby*nlayer
    for i in range(Ngrains):
        rand = int(np.random.random()*len(lballast))
        body = pre.rigidPolyhedron(model = model, material=mat, color=color, generation_type='full', vertices=lballast[rand][0], faces=lballast[rand][1])
        r = np.random.random(3)*np.pi
        body.rotate(phi=r[0],theta=r[1],psi=r[2])
        body.translate(dx=x[i]+0.1, dy=y[i], dz=z[i])
        #body.imposeDrivenDof(dofty='vloc', component=[1,2,3,4,5,6])
        #body.imposeDrivenDof(dofty='force', description='predefined', component=3, ct=-4.9*0.0005)
        i+=1
        bodies.addAvatar(body)
    return bodies, z

def plate_definition(dict, mod, mat, dt, time):
   bodies = pre.avatars()
   for k, v in dict.items():
      # the key is the name of the plate
      plate = pre.avatar(dimension=3)
      plate.addBulk(pre.rigid3d())
      # add node
      plate.addNode(pre.node(np.array([0.,0.,0.]), 1))
      # define Groups
      plate.defineGroups()
      plate.defineModel(model=mod)
      plate.defineMaterial(material=mat)
      # define contactors with respect to dic value
      color = 'VERTx' if 'color' not in v else v['color']
      plate.addContactors('PLANx', color=color, axe1=v['axe1'], axe2=v['axe2'], axe3=v['axe3'], group='all')
      plate.computeRigidProperties()
      if 'rotate_Ax' in v:
         plate.rotate(center=plate.nodes[1].coor,**v['rotate_Ax'])
      plate.translate(dx=v['x'], dy=v['y'], dz=v['z'])
      plate.rotate(center=plate.nodes[1].coor,**v['rotate']) if 'rotate' in v else None

      plate.imposeDrivenDof(dofty='vlocy',**v['imposeDrivenDof'])
      if 'DrivenDof' in v:
        def imposedv(t):
          if t > time -1 :
            return vx
          else:
            # return gravity
             return vinit * t
          
        vx = v['DrivenDof']['vx']
        vinit = 0. if 'vinit' not in v['DrivenDof'] else v['DrivenDof']['vinit']
        fname = v['DrivenDof']['name'] + '.dat'
        if 'start' not in v['DrivenDof']:start = 0.
        else: start = v['DrivenDof']['start']
        pre.writeEvolution(f=imposedv, instants=np.linspace(start,time, int(1/dt)), path=v['DrivenDof']['path'] + 'DATBOX/',name=fname)
        plate.imposeDrivenDof(component=v['DrivenDof']['component'], description='evolution',dofty='vlocy', evolutionFile=fname)
  
      if 'pullup' in v:
         # previously the body fell under gravity now pull it up
         def drivenForce(t):
            if t > time - 1: 
               #return 600
               return -v['pullup']['acc'] *4
            else:
               return -v['pullup']['acc']
            
         pre.writeEvolution(f=drivenForce, instants=np.linspace(0,time,int(1/dt)), path=v['pullup']['path'] + 'DATBOX/',name=v['pullup']['name'] + '.dat')
         plate.imposeDrivenDof(component=v['pullup']['component'], description='evolution',dofty='force', evolutionFile=v['pullup']['name'] + '.dat')
      bodies.addAvatar(plate)
      if 'pullin' in v:
          def drivenForce(t):
              if t> 2.5 and t<3.4:
                return v['pullin']['acc']
              elif t> 3.5 and t<10:
                return -v['pullin']['acc']
              else:
                return 0
              
          pre.writeEvolution(f=drivenForce, instants=np.linspace(0,time,int(1/dt)), path=v['pullin']['path'] + 'DATBOX/',name=v['pullin']['name'] + '.dat')
          plate.imposeDrivenDof(component=v['pullin']['component'], description='evolution',dofty='force', evolutionFile=v['pullin']['name'] + '.dat')
   return bodies





        


def get_lbody(file):
  f=open(file,'r')
  nbbody= int(f.readline())
  lbody=[]
  for nb in range(nbbody):
    nlv = int( f.readline())
    vertex=[]
    for i in range(nlv):
      vertex.append( f.readline())
    vertex = np.loadtxt(vertex)
    vertex-= np.mean( vertex,axis=0)
    vertex/=1.	
    face=[]
    nlf = int(f.readline())
    for i in range(nlf):
      face.append( f.readline())
    face  = np.loadtxt(face,dtype='int')
    endbody= f.readline()
    lbody.append( (vertex, face))
  f.close()
  return lbody
