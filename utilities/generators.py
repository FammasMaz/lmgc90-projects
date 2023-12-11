from pylmgc90 import pre
import math
import numpy as np

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

def ballast_generator(nb_particles, Rmin, Rmax, Px, Py, Pz, layers, mat, mod, ptype, min_vert, max_vert, gen_type='box', seed=43, color='BLUEx'):
   bodies = pre.avatars()
   total_particles = 0
   for j in range(len(layers)):
      radii = pre.granulo_Random(nb_particles, Rmin, Rmax, seed=seed)
      #if gen_type == 'box':
      [nb_rem,coors] = pre.depositInBox3D(radii, Px, Py*layers[j], Pz)
      #else:
         #coors = pre.cubicLattice3D(int(Px/Rmax), int(Py*layers[j]/Rmax), 1, Rmax*2)
         #nb_rem = len(coors)//3
      for i in range(nb_rem):
          if ptype == 'POLYR':
              body = pre.rigidPolyhedron(radius=radii[i], center=coors[3*i : 3*(i+1)], nb_vertices=np.random.randint(min_vert, max_vert), generation_type='random',model=mod,
                                material=mat, color=color)
          else:
              body = pre.rigidSphere(r=radii[i],center=coors[3*i:3*(i+1)],model=mod,
                                material=mat,color=color)
          body.translate(dz=Pz*(len(layers)-j) + Rmax*4.)
          bodies.addAvatar(body)
      total_particles += nb_rem
   return bodies, total_particles

def wall_generator(wall_length, wall_height, wall_width, wall_material, wall_model, rot_axis=None,alpha=math.pi/2., wall_offset=None, wall_color='WALLx'):
        bodies = pre.avatars()
        wall = pre.rigidPlan(axe1=wall_length/2.,axe2=wall_height/2.,axe3=wall_width/2.,center=[0.,0.,0.],
                       material=wall_material,model=wall_model,color=wall_color)
        [wall.translate(dx=wall_offset[0], dy=wall_offset[1], dz=wall_offset[2]) if wall_offset is not None else None]
        [wall.rotate(description='axis',alpha=alpha,axis=rot_axis,center=wall.nodes[1].coor) if rot_axis is not None else None]
        bodies.addAvatar(wall)
        return bodies