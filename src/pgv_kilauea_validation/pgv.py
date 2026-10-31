"""Directional PGV, deterministic null correction, and GeoTIFF export."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

from . import _pgv_core as core
from .io import atomic_path,digest,write_json


def read_pair(config,row,verify_hashes=True):
    files=[config.input_path(row,k) for k in ['phase','coherence']]
    if verify_hashes:
        for key,path in zip(['phase','coherence'],files):
            if digest(path)!=row[key+'_sha256']:raise ValueError(f'Input hash mismatch: {path}')
    with rasterio.open(files[0]) as ds,rasterio.open(files[1]) as co:
        if (ds.crs,ds.transform,ds.shape)!=(co.crs,co.transform,co.shape):
            raise ValueError('Phase/coherence grid mismatch: '+row['pair_id'])
        if ds.crs is None or ds.crs.to_epsg()!=4326:raise ValueError('Expected EPSG:4326 phase grid')
        phase=ds.read(1);coherence=co.read(1)
        phase=np.where((phase < -9000)|(ds.read_masks(1)==0),np.nan,phase)
        coherence=np.where(co.read_masks(1)>0,coherence,np.nan)
        if np.any(np.abs(phase[np.isfinite(phase)])>np.pi+1e-4):raise ValueError('Expected wrapped phase in radians')
        finite=coherence[np.isfinite(coherence)]
        if np.any((finite<0)|(finite>1)):raise ValueError('Coherence outside [0,1]')
        return phase,coherence,ds.transform.to_gdal()


def vote_settings(config,smoothing=None):
    p=config.values['pgv'];spacing=p['vote_spacing_m'];extent=p['extent_m'];n=int(2*extent/spacing)+1
    return dict(vote_sp=spacing,xn=-extent,xx=extent,yn=-extent,yx=extent,vnx=n,vny=n,
                ray_min=p['ray_min_m'],ray_max=p['ray_max_m'],n_ray_steps=p['ray_steps'],ray_spread=0.,
                smooth_sigma_m=p['smooth_sigma_m'] if smoothing is None else smoothing)


def observations(config,row):
    phase,coh,gt=read_pair(config,row);p=config.values['pgv'];ref=config.values['reference']
    dlat,dlon=core.deg2m(ref['lat'])
    _,_,direction=core.compute_wrapped_gradient(phase,coh,p['coherence_threshold'],gt,dlon,dlat,p['gradient_step_px'])
    circle=core.build_caldera_mask(gt,*phase.shape,ref['lat'],ref['lon'],p['radius_m'])
    valid=circle&(coh>=p['coherence_threshold'])&np.isfinite(direction)
    rr,cc=np.nonzero(valid);n=len(rr)
    if not n:raise ValueError('No valid gradient observations: '+row['pair_id'])
    stride=max(1,n//p['target_vote_points']);rr,cc=rr[::stride],cc[::stride]
    lon,lat=core.pixel_to_lonlat(gt,rr,cc);x=(lon-ref['lon'])*dlon;y=(lat-ref['lat'])*dlat
    weights=coh[rr,cc].astype(float)
    if p['weight']=='coherence_squared':weights=weights**2
    elif p['weight']=='uniform':weights=np.ones(len(rr))
    return dict(x=x,y=y,directions=direction[rr,cc],weights=weights,phase=phase,coherence=coh,gt=gt,
                n_valid=n,n_domain=int(circle.sum()),n_sampled=len(rr))


def compute(config,row,raw=False):
    obs=observations(config,row);seed=int(row['null_seed'])
    if not 0<=seed<2**32:raise ValueError('Invalid null seed')
    null_dirs=np.random.RandomState(seed).uniform(-np.pi,np.pi,obs['n_sampled'])
    kwargs=vote_settings(config,smoothing=0. if raw else None)
    actual=np.array(core.do_vote_directional(obs['x'],obs['y'],obs['directions'],weights=obs['weights'],**kwargs))
    null=np.array(core.do_vote_directional(obs['x'],obs['y'],null_dirs,weights=obs['weights'],**kwargs))
    result=actual-null
    if not np.isfinite(result).all():raise ValueError('Nonfinite PGV result')
    n=kwargs['vnx'];p=config.values['pgv']
    obs.update(actual=actual,null=null,result=result,x_m=-p['extent_m']+np.arange(n)*p['vote_spacing_m'])
    obs['y_m']=obs['x_m'].copy()
    return obs


def signature(config,row):
    return dict(pgv=config.values['pgv'],reference=config.values['reference'],null_seed=int(row['null_seed']),
                phase_sha256=row['phase_sha256'],coherence_sha256=row['coherence_sha256'],
                core_sha256=digest(Path(core.__file__)),adapter_sha256=digest(Path(__file__)))


def run_pair(config,row,overwrite=False):
    folder=config.output/'pgv';target=folder/(row['pair_id']+'.npz');meta=target.with_suffix('.json')
    wanted=signature(config,row)
    if target.exists() and meta.exists() and not overwrite:
        old=json.loads(meta.read_text())
        if old['signature']!=wanted:raise ValueError('Existing PGV has different settings; choose a new output directory or --overwrite')
        if old['output_sha256']!=digest(target):raise ValueError('Existing PGV artifact was modified')
        return old
    obs=compute(config,row)
    fraction=obs['n_valid']/obs['n_domain']
    if abs(fraction-float(row['valid_fraction']))>1e-12:
        # Scientific setting changes may change valid pixels, but never silently retain an old cohort.
        raise ValueError('Valid-gradient fraction differs from audited manifest: '+row['pair_id'])
    with atomic_path(target) as p:
        with p.open('wb') as f:
            np.savez_compressed(f,fwd=obs['result'][0],bwd=obs['result'][1],x_m=obs['x_m'],y_m=obs['y_m'],
                                n_valid_pts=obs['n_valid'],n_mask_pts=obs['n_domain'],
                                valid_fraction=fraction,null_seed=int(row['null_seed']))
    record=dict(pair_id=row['pair_id'],track=row['track'],analysis_class=row['analysis_class'],
                n_valid_pts=obs['n_valid'],n_sampled=obs['n_sampled'],valid_fraction=fraction,
                signature=wanted,output_sha256=digest(target))
    write_json(meta,record);return record


def load_maps(config,row):
    path=config.output/'pgv'/(row['pair_id']+'.npz')
    record=json.loads(path.with_suffix('.json').read_text())
    if record['signature']!=signature(config,row) or record['output_sha256']!=digest(path):
        raise ValueError('PGV output signature/hash mismatch: '+row['pair_id'])
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in ['fwd','bwd','x_m','y_m']}


def export_geotiffs(config,row):
    maps=load_maps(config,row);x,y=maps['x_m'],maps['y_m'];ref=config.values['reference']
    dlat,dlon=core.deg2m(ref['lat']);spacing=config.values['pgv']['vote_spacing_m']
    transform=from_origin(ref['lon']+(x[0]-spacing/2)/dlon,ref['lat']+(y[-1]+spacing/2)/dlat,spacing/dlon,spacing/dlat)
    for key in ['fwd','bwd']:
        path=config.output/'geotiff'/(row['pair_id']+'_'+key+'.tif')
        with atomic_path(path) as tmp:
            with rasterio.open(tmp,'w',driver='GTiff',height=len(y),width=len(x),count=1,dtype='float32',
                               crs='EPSG:4326',transform=transform,compress='deflate',nodata=np.nan) as ds:
                ds.write(maps[key][::-1].astype('float32'),1)
                ds.update_tags(quantity='Null-corrected, Gaussian-smoothed directional votes',direction=key)
