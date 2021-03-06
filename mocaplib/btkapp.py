"""
MIT License

Copyright (c) 2020 Moon Ki Jung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__author__ = "Moon Ki Jung"

import os
import sys
import numpy as np
import btk
#%%

def open_c3d(f_path=None):
    if f_path is None: return None
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(f_path)
    reader.Update()
    acq = reader.GetOutput()
    return acq

def save_c3d(acq, f_path):
    if f_path is None: return None
    writer = btk.btkAcquisitionFileWriter()
    writer.SetInput(acq)
    writer.SetFilename(f_path)
    return writer.Update()

def get_first_frame(acq):
    return np.int32(acq.GetFirstFrame())

def get_last_frame(acq):
    return np.int32(acq.GetLastFrame())

def get_num_frames(acq):
    return np.int32(acq.GetPointFrameNumber())

def get_video_fps(acq):
    return np.float32(acq.GetPointFrequency())

def get_analog_video_ratio(acq):
    return np.int32(acq.GetNumberAnalogSamplePerFrame())

def get_analog_fps(acq):
    return np.float32(acq.GetAnalogFrequency())

def get_video_frames(acq):
    first_fr = get_first_frame(acq)
    last_fr = get_last_frame(acq)
    n_frs = get_num_frames(acq)
    frs = np.linspace(first_fr, last_fr, n_frs, dtype=np.int32)
    return frs

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
    
def get_dict_metadata(acq, desc=False):
    dict_md = {}
    md_root = acq.GetMetaData()
    get_sub_dict_metadata(md_root, dict_md, desc)
    return dict_md['ROOT']

def get_sub_dict_metadata(md, dict_parent, desc=False):
    md_label = md.GetLabel()
    md_desc = md.GetDescription()
    if md.HasInfo():
        md_info = md.GetInfo()
        md_dim = md_info.GetDimension(0)
        md_dims = md_info.GetDimensions()
        md_fmt = md_info.GetFormatAsString()
        if md_fmt == 'Byte' or md_fmt == 'Integer':
            md_val = np.array(md_info.ToInt(), dtype=np.int32)
        if md_fmt == 'Real':
            md_val = np.array(md_info.ToDouble(), dtype=np.float32)
        if md_fmt == 'Char':
            md_val = np.array([x.strip() for x in md_info.ToString()], dtype=str)
        if md_fmt == 'Char':
            if len(md_dims) <= 1:
                md_val = md_val.item()
            else:
                md_val = np.reshape(md_val, md_dims[::-1][:-1])
        else:
            if md_dim == 0:
                md_val = md_val.item()
            else:
                md_val = np.reshape(md_val, md_dims[::-1])
        if desc:
            dict_parent.update({md_label: {}})
            dict_parent[md_label].update({'VAL': md_val})
            dict_parent[md_label].update({'DESC': md_desc})
        else:
            dict_parent.update({md_label: md_val})
    else:
        dict_parent.update({md_label:{}})
    if md.HasChildren():
        n_child = md.GetChildNumber()
        for i in range(n_child):
            md_child = md.GetChild(i)
            get_sub_dict_metadata(md_child, dict_parent[md_label], desc)
    
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

def get_fp_info(acq):
    fpe = btk.btkForcePlatformsExtractor()
    fpe.SetInput(acq)
    fpe.Update()
    fpc = fpe.GetOutput()
    n_fp = fpc.GetItemNumber()
    dict_fp = {}
    for i in range(n_fp):
        dict_fp.update({i:{}})
        fp = fpc.GetItem(i)
        type = fp.GetType()
        dict_fp[i].update({'TYPE': type})
        n_chs = fp.GetChannelNumber()
        dict_fp[i].update({'VALUES':{}})
        labels = []
        for j in range(n_chs):
            ch = fp.GetChannel(j)
            ch_name = ch.GetLabel()
            labels.append(ch_name)
            # dict_fp[i]['channel'].update({ch_name:{}})
            ch_val = ch.GetValues()
            dict_fp[i]['VALUES'].update({ch_name: np.asarray(np.squeeze(ch_val), dtype=np.float32)})
        dict_fp[i].update({'LABELS': labels})
        corners = fp.GetCorners()
        dict_fp[i].update({'CORNERS': np.asarray(corners.T, dtype=np.float32)})
        origin = fp.GetOrigin()
        dict_fp[i].update({'ORIGIN': np.asarray(np.squeeze(origin.T), dtype=np.float32)})
        cal_matrix = fp.GetCalMatrix()
        dict_fp[i].update({'CAL_MATRIX': np.asarray(cal_matrix.T, dtype=np.float32)})
    return dict_fp

def add_metadata(acq, parent_name, name, data, desc=None):
    md_root = acq.GetMetaData()
    grp_names = []
    n_child = md_root.GetChildNumber()
    for i in range(n_child):
        grp_names.append(md_root.GetChild(i).GetLabel())
    if parent_name not in grp_names:
        return False
    md_parent = md_root.FindChild(parent_name).value()
    btk.btkMetaDataCreateChild(md_parent, name, data)
    md_child = md_parent.FindChild(name).value()
    if desc is None or type(desc) is not str:
        child_desc = ''
    else:
        child_desc = desc
    md_child.SetDescription(child_desc)
    return True

def change_point_name(acq, old_name, new_name):
    pt = acq.FindPoint(old_name).value()
    pt.SetLabel(new_name)
    