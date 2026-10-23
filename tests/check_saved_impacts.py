import ast, numpy as np, struct
from pathlib import Path
package=Path(__file__).resolve().parents[1]
root=package/'rfa_model'
ns={'np':np,'unit':lambda v:np.asarray(v)/np.linalg.norm(v)}
def functions(file,names):
 tree=ast.parse((root/file).read_text()); nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];exec(compile(ast.Module(body=nodes,type_ignores=[]),file,'exec'),ns)
functions('collisions.py',['_is_finite_vec','canonical_collision_owner_name','add_owner_metadata','make_sample_plane_geometry','_validated_sample_geometry','sample_plane_coordinates','sample_point_is_in_bounds','segment_hits_sample_plane','nearest_hit'])
b=(package/'tests/fixtures/aligned_sample_89deg(1).stl').read_bytes();rec=np.frombuffer(b,dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]),offset=84,count=struct.unpack_from('<I',b,80)[0]);tri=rec['v'].astype(float)
ns['OWNER_ID_BY_NAME']={'sample':1}
def segment(p0,p1,*args,exclude_owners=None,**kwargs):
 if exclude_owners and 'sample' in exclude_owners:return None
 dr=p1-p0;e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];h=np.cross(dr,e2);det=np.einsum('ij,ij->i',e1,h);valid=abs(det)>1e-20;inv=np.zeros_like(det);inv[valid]=1/det[valid];s=p0-tri[:,0];a=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1);bb=q@dr*inv;t=np.einsum('ij,ij->i',e2,q)*inv;ok=valid&(a>=-1e-8)&(bb>=-1e-8)&(a+bb<=1+1e-8)&(t>1e-9)&(t<=1+1e-8)
 if not ok.any():return None
 ids=np.flatnonzero(ok);f=ids[np.argmin(t[ids])];return {'kind':'stl','owner':'sample','owner_name':'sample','normal':rec['n'][f].astype(float),'location':p0+t[f]*dr,'t':float(t[f]),'distance':float(t[f]*np.linalg.norm(dr))}
ns.update(first_segment_hit=segment,e_charge=1.602176634e-19,m_e=9.1093837e-31,E_at_point=lambda *x:np.zeros(3),classify_grid_point=lambda *x:{'status':'free'},kinetic_energy_eV_from_velocity=lambda v:9.1093837e-31*np.dot(v,v)/(2*1.602176634e-19))
functions('primary.py',['_stl_owner_is_sample','_finalize_sample_stl_hit','fly_primary_to_sample'])
geom=ns['make_sample_plane_geometry']((-.00635,.00635),(-.004375,.004375),89);geom['center']=np.array([.000119981609,-.00000209428678,0])
z=np.load(next((package/'tests/fixtures').glob('*trajectories(1).npz')))
for j,idx in enumerate(z['primary_primary_index']):
 p=z['primary_points'][z['primary_offsets'][j]:z['primary_offsets'][j+1]];v=z['primary_velocities'][z['primary_offsets'][j]:z['primary_offsets'][j+1]]
 crossing=next((k for k in range(len(p)-1) if segment(p[k],p[k+1]) is not None),None);assert crossing is not None
 k=crossing;dt=np.linalg.norm(p[k+1]-p[k])/np.linalg.norm(v[k]);r=ns['fly_primary_to_sample'](p[k],v[k],{},None,None,None,None,None,None,None,dt=dt*1.01,max_steps=3,adaptive_dt=False,sample_geometry=geom)
 assert r['reason']=='hit_sample',(idx,r)
 face=r['hit_info'].get('sample_face','front');expected='side' if j<5 else 'front';assert face==expected,(idx,face)
 print(idx,'correctly stopped:',face)
# Verify geometric inference agrees with STL, and rejects 85/89 mismatch.
from types import SimpleNamespace
mesh=SimpleNamespace(faces=np.arange(len(tri)*3).reshape(-1,3),vertices=tri.reshape(-1,3),face_normals=rec['n'].astype(float),area_faces=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2)
ns['canonical_surface_name']=lambda x:x
functions('cascade.py',['_patch_sample_geometry_extents','infer_sample_front_face_geometry_from_stl','estimate_surface_normal'])
g=ns['infer_sample_front_face_geometry_from_stl'](mesh,['sample']*12,89,(-.00635,.00635),(-.004375,.004375));assert g['front_face_triangle_count']==2;assert np.allclose(g['center'],geom['center'],atol=1e-9)
try:ns['infer_sample_front_face_geometry_from_stl'](mesh,['sample']*12,85,(-.00635,.00635),(-.004375,.004375))
except ValueError:pass
else:raise AssertionError('angle mismatch not rejected')
print('Geometry and mismatch guard passed; all ten collision cases passed.')
