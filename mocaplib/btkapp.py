import os
import sys
import numpy as np
import btk
#%%

def get_acq(f_path=None):
    if f_path is None: return None
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(f_path)
    reader.Update()
    acq = reader.GetOutput()
    return acq

def get_point_names(acq, tgt_types=None):
    pt_names = []
    pt_types = None if tgt_types is None else set()
    if tgt_types is not None:
        for type in tgt_types:
            if type == 'Angle':
                pt_types.add(btk.btkPoint.Angle)
            elif type == 'Force':
                pt_types.add(btk.btkPoint.Force)
            elif type == 'Marker':
                pt_types.add(btk.btkPoint.Marker)
            elif type == 'Moment':
                pt_types.add(btk.btkPoint.Moment)
            elif type == 'Power':
                pt_types.add(btk.btkPoint.Power)
            elif type == 'Reaction':
                pt_types.add(btk.btkPoint.Reaction)
            elif type == 'Scalar':
                pt_types.append(btk.btkPoint.Scalar)
    for pt in btk.Iterate(acq.GetPoints()):
        if pt_types is not None and pt.GetType() not in pt_types: continue
        pt_names.append(pt.GetLabel())
    return pt_names

def get_analog_names(acq):
    sig_names = []
    for sig in btk.Iterate(acq.GetAnalogs()):
        sig_names.append(sig.GetLabel())
    return sig_names

def get_dict_metadata(acq):
    dict_md = {}
    md_root = acq.GetMetaData()
    get_sub_dict_metadata(md_root, dict_md)
    return dict_md['ROOT']

def get_sub_dict_metadata(md, dict_parent):
    md_label = md.GetLabel()
    if md.HasInfo():
        md_info = md.GetInfo()
        # md_dim = md_info.GetDimension(0)
        # md_dims = md_info.GetDimensions()
        # md_dims_prod = md_info.GetDimensionsProduct()
        md_fmt = md_info.GetFormatAsString()
        if md_fmt == 'Byte' or md_fmt == 'Integer':
            md_val = np.squeeze(np.array(md_info.ToInt(), dtype=np.int32))
            if len(md_val.shape)==0 or (len(md_val.shape)==1 and md_val.shape[0]==1): md_val = np.int32(md_val.item())
        if md_fmt == 'Real':
            md_val = np.squeeze(np.array(md_info.ToDouble(), dtype=np.float32))
            if len(md_val.shape)==0 or (len(md_val.shape)==1 and md_val.shape[0]==1): md_val = np.float32(md_val.item())
        if md_fmt == 'Char':
            md_val = np.squeeze(np.array([x.strip() for x in md_info.ToString()], dtype=str))
            if len(md_val.shape)==0 or (len(md_val.shape)==1 and md_val.shape[0]==1): md_val = md_val.item()
        dict_parent.update({md_label: md_val})
    else:
        dict_parent.update({md_label:{}})
    if md.HasChildren():
        n_child = md.GetChildNumber()
        for i in range(n_child):
            md_child = md.GetChild(i)
            get_sub_dict_metadata(md_child, dict_parent[md_label])
        

def get_dict_events(acq):
    if acq.IsEmptyEvent(): return None
    dict_events = {}
    for ev in btk.Iterate(acq.GetEvents()):
        name = ev.GetLabel()
        context = ev.GetContext()
        desc = ev.GetDescription()
        fr = ev.GetFrame()
        dict_events.update({fr: {}})
        dict_events[fr].update({'FRAME': fr})
        dict_events[fr].update({'LABEL': name})
        dict_events[fr].update({'CONTEXT': context})
        dict_events[fr].update({'DESCRIPTION': desc})
    return dict_events

def get_dict_points(acq, blocked_nan=False, resid=False, tgt_types=None):
    if acq.IsEmptyPoint(): return None
    pt_types = None if tgt_types is None else set()
    if tgt_types is not None:
        for type in tgt_types:
            if type == 'Angle':
                pt_types.add(btk.btkPoint.Angle)
            elif type == 'Force':
                pt_types.add(btk.btkPoint.Force)
            elif type == 'Marker':
                pt_types.add(btk.btkPoint.Marker)
            elif type == 'Moment':
                pt_types.add(btk.btkPoint.Moment)
            elif type == 'Power':
                pt_types.add(btk.btkPoint.Power)
            elif type == 'Reaction':
                pt_types.add(btk.btkPoint.Reaction)
            elif type == 'Scalar':
                pt_types.append(btk.btkPoint.Scalar)
    pt_labels = []
    pt_descs = []
    dict_pts = {}
    dict_pts.update({'DATA': {}})
    dict_pts['DATA'].update({'POS': {}})
    if resid: dict_pts['DATA'].update({'RESID': {}})
    for pt in btk.Iterate(acq.GetPoints()):
        if pt_types is not None and pt.GetType() not in pt_types: continue
        pt_name = pt.GetLabel()
        pt_pos = pt.GetValues()
        if blocked_nan or resid:
            pt_resid = pt.GetResiduals().flatten()
        if blocked_nan:
            pt_null_masks = np.where(np.isclose(pt_resid, -1), True, False)
            pt_pos[pt_null_masks,:] = np.nan
        pt_desc = pt.GetDescription()
        pt_labels.append(pt_name)
        pt_descs.append(pt_desc)
        dict_pts['DATA']['POS'].update({pt_name: np.asarray(pt_pos, dtype=np.float32)})
        if resid:
            dict_pts['DATA']['RESID'].update({pt_name: np.asarray(pt_resid, dtype=np.float32)})
    dict_pts.update({'LABELS': np.array(pt_labels, dtype=str)})    
    dict_pts.update({'DESCRIPTIONS': np.array(pt_descs, dtype=str)})
    dict_pts.update({'UNITS': acq.GetPointUnit()})    
    dict_pts.update({'RATE': np.float32(acq.GetPointFrequency())})
    dict_pts.update({'FRAME': np.linspace(acq.GetFirstFrame(), acq.GetLastFrame(), acq.GetPointFrameNumber(), dtype=np.int32)})
    return dict_pts

def get_dict_analogs(acq):
    if acq.IsEmptyAnalog(): return None
    sig_labels = []
    sig_descs = []
    sig_units = []
    sig_offset = []
    sig_scale = []
    sig_gain = []
    dict_sigs = {}
    dict_sigs.update({'DATA': {}})
    for sig in btk.Iterate(acq.GetAnalogs()):
        name = sig.GetLabel()
        offset = sig.GetOffset()
        scale = sig.GetScale()
        unit = sig.GetUnit()
        desc = sig.GetDescription()
        gain = sig.GetGain()
        sig_labels.append(name)
        sig_descs.append(desc)
        sig_units.append(unit)
        sig_offset.append(offset)
        sig_scale.append(scale)
        sig_gain.append(gain)
        val = np.asarray(sig.GetValues().flatten(), dtype=np.float32)
        dict_sigs['DATA'].update({name: val})
    dict_sigs.update({'LABELS': np.array(sig_labels, dtype=str)})
    dict_sigs.update({'DESCRIPTIONS': np.array(sig_descs, dtype=str)})
    dict_sigs.update({'UNITS': np.array(sig_units, dtype=str)})
    dict_sigs.update({'RATE': np.float32(acq.GetAnalogFrequency())})
    dict_sigs.update({'OFFSET': np.array(sig_offset, dtype=np.float32)})
    dict_sigs.update({'SCALE': np.array(sig_scale, dtype=np.float32)})
    dict_sigs.update({'GAIN': np.array(sig_gain, dtype=np.int32)})
    dict_sigs.update({'RESOLUTION': np.int32(acq.GetAnalogResolution())})
    return dict_sigs