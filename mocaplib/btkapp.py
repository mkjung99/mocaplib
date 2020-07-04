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

def get_dict_events(acq):
    num_events = acq.GetEventNumber()
    dict_events = {}
    for i in range(num_events):
        ev = acq.GetEvent(i)
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

def get_dict_markers(acq):
    pts = acq.GetPoints()
    num_pts = pts.GetItemNumber()
    mkr_labels = []
    mkr_descs = []
    dict_mkrs = {}
    dict_mkrs.update({'DATA': {}})
    dict_mkrs['DATA'].update({'POS': {}})
    dict_mkrs['DATA'].update({'RESID': {}})
    for i in range(num_pts):
        pt = pts.GetItem(i)
        name = pt.GetLabel()
        pos = pt.GetValues()
        resid = pt.GetResiduals().flatten()
        desc = pt.GetDescription()
        mkr_labels.append(name)
        mkr_descs.append(desc)
        dict_mkrs['DATA']['POS'].update({name: np.asarray(pos, dtype=np.float32)})
        dict_mkrs['DATA']['RESID'].update({name: np.asarray(resid, dtype=np.float32)})
    dict_mkrs.update({'LABELS': np.array(mkr_labels, dtype=str)})    
    dict_mkrs.update({'DESCRIPTIONS': np.array(mkr_descs, dtype=str)})
    dict_mkrs.update({'UNITS': acq.GetPointUnit()})    
    dict_mkrs.update({'RATE': np.float32(acq.GetPointFrequency())})
    dict_mkrs.update({'FRAMES': np.linspace(acq.GetFirstFrame(), acq.GetLastFrame(), acq.GetPointFrameNumber(), dtype=np.int32)})
    return dict_mkrs

def get_dict_analogs(acq):
    sigs = acq.GetAnalogs()
    num_sigs = sigs.GetItemNumber()
    sig_labels = []
    sig_descs = []
    sig_units = []
    sig_offset = []
    sig_scale = []
    sig_gain = []
    dict_sigs = {}
    dict_sigs.update({'DATA': {}})
    for i in range(num_sigs):
        sig = sigs.GetItem(i)
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