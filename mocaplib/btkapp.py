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
from scipy.signal import butter, filtfilt
import re
import btk
#%%
def filt_bw_bp(data, fc_low, fc_high, fs, order=2):
    nyq = 0.5 * fs
    low = fc_low / nyq
    high = fc_high / nyq
    b, a = butter(order, [low, high], analog=False, btype='bandpass', output='ba')
    axis = -1 if len(data.shape)==1 else 0
    y = filtfilt(b, a, data, axis, padtype='odd', padlen=3*(max(len(b),len(a))-1))
    return y

def filt_bw_bs(data, fc_low, fc_high, fs, order=2):
    nyq = 0.5 * fs
    low = fc_low / nyq
    high = fc_high / nyq
    b, a = butter(order, [low, high], analog=False, btype='bandstop', output='ba')
    axis = -1 if len(data.shape)==1 else 0
    y = filtfilt(b, a, data, axis, padtype='odd', padlen=3*(max(len(b),len(a))-1))
    return y

def filt_bw_lp(data, fc_low, fs, order=2):
    nyq = 0.5 * fs
    low = fc_low / nyq
    b, a = butter(order, low, analog=False, btype='lowpass', output='ba')
    axis = -1 if len(data.shape)==1 else 0
    y = filtfilt(b, a, data, axis, padtype='odd', padlen=3*(max(len(b),len(a))-1))
    return y

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
                if md_val.shape[0] == 0:
                    md_val = ''
                else:
                    md_val = md_val.item()
            else:
                md_val = np.reshape(md_val, md_dims[::-1])
        # special handling for 'FORCE_PLATFORM:CAL_MATRIX' parameter
        if md_label=='CAL_MATRIX' and len(md_dims)==3 and (md_dims[0]==6 and md_dims[1]==6):
            md_val = np.transpose(md_val, (0,2,1))
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

def get_fp_output(acq, threshold=0.0, filt_fc=None, filt_order=2, cop_nan_to_num=True):
    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(acq)
    pfe.Update()
    pfc = pfe.GetOutput()
    if pfc.IsEmpty():
        return None
    point_unit = acq.GetPointUnit()
    point_scale = 1.0 if point_unit=='m' else 0.001
    analog_fps = acq.GetAnalogFrequency()
    rgx_fp = re.compile(r'\S*(\d*[FMP]\d*[XxYyZz]\d*)')
    fp_output = {}
    fp_idx = 0
    for fp in btk.Iterate(pfc):
        fp_type = fp.GetType()
        # force plate location info
        fp_org_raw = np.squeeze(fp.GetOrigin())*point_scale
        fp_z_check = -1.0 if fp_org_raw[2]>0 else 1.0
        if fp_type == 1:
            o_x = 0.0
            o_y = 0.0
            o_z = (-1.0)*fp_org_raw[2]*fp_z_check
        elif fp_type in [2, 4]:
            o_x = (-1.0)*fp_org_raw[0]*fp_z_check
            o_y = (-1.0)*fp_org_raw[1]*fp_z_check
            o_z = (-1.0)*fp_org_raw[2]*fp_z_check
        elif fp_type == 3:
            o_x = 0.0
            o_y = 0.0
            o_z = (-1.0)*fp_org_raw[2]*fp_z_check
            fp_len_a = np.abs(fp_org_raw[0])
            fp_len_b = np.abs(fp_org_raw[1])
        fp_corners = fp.GetCorners().T*point_scale
        # fp_corners[0] #(+x, +y)
        # fp_corners[1] #(-x, +y)
        # fp_corners[2] #(-x, -y)
        # fp_corners[3] #(+x, -y)        
        fp_cen = np.mean(fp_corners, axis=0)
        fp_len_x = (np.linalg.norm(fp_corners[0]-fp_corners[1])+np.linalg.norm(fp_corners[3]-fp_corners[2]))*0.5
        fp_len_y = (np.linalg.norm(fp_corners[0]-fp_corners[3])+np.linalg.norm(fp_corners[1]-fp_corners[2]))*0.5
        fp_p0 = fp_cen
        fp_p1 = 0.5*(fp_corners[0]+fp_corners[3])
        fp_p2 = 0.5*(fp_corners[0]+fp_corners[1])
        fp_v0 = fp_p1-fp_p0
        fp_v1 = fp_p2-fp_p0
        fp_v0_u = fp_v0/np.linalg.norm(fp_v0)
        fp_v1_u = fp_v1/np.linalg.norm(fp_v1)
        fp_v2 = np.cross(fp_v0_u, fp_v1_u)
        fp_v2_u = fp_v2/np.linalg.norm(fp_v2)
        fp_v_z = fp_v2_u
        fp_v_x = fp_v0_u
        fp_v_y = np.cross(fp_v_z, fp_v_x)
        fp_rot_mat = np.column_stack([fp_v_x, fp_v_y, fp_v_z])
        # force plate force/moment info
        fp_data = {}
        ch_data = {}
        ch_scale = {}
        # for ch in btk.Iterate(fp.GetChannels()):
        fp_cnt_chs = fp.GetChannelNumber()
        if filt_fc is None:
            filt_fcs = [None]*fp_cnt_chs
        elif type(filt_fc) in [int, float]:
            filt_fcs = [float(filt_fc)]*fp_cnt_chs
        elif type(filt_fc) in [list, tuple]:
            filt_fcs = list(filt_fc)
        elif type(filt_fc)==np.ndarray:
            if len(filt_fc)==1:
                filt_fcs = [filt_fc.item()]*fp_cnt_chs
            else:
                filt_fcs = filt_fc.tolist()
        for ch_idx in range(fp_cnt_chs):
            ch_name = fp.GetChannel(ch_idx).GetLabel()
            ch = acq.GetAnalog(ch_name)
            fm_name =  str.upper(rgx_fp.findall(ch.GetLabel())[0])
            # assign channel names
            if fp_type == 1:
                # assume that the order of input analog channels are as follows:
                # 'FX', 'FY', 'FZ', 'PX', 'PY', 'TZ'
                label = ['FX', 'FY', 'FZ', 'PX', 'PY', 'TZ'][ch_idx]            
            elif fp_type in [2, 4]:
                label = re.sub(r'\d', r'', fm_name)
            elif fp_type == 3:
                # assume that the order of input analog channels are as follows:
                # 'FX12', 'FX34', 'FY14', 'FY23', 'FZ1', 'FZ2', 'FZ3', 'FZ4'
                label = ['FX12', 'FX34', 'FY14', 'FY23', 'FZ1', 'FZ2', 'FZ3', 'FZ4'][ch_idx]
            # assign channel scale factors
            if fp_type == 1:
                if label.startswith('F'):
                    # assume that the force unit is 'N'
                    ch_scale[label] = 1.0
                elif label.startswith('T'):
                    # assume that the torque unit is 'Nmm'
                    ch_scale[label] = 0.001
                    if ch.GetUnit()=='Nm': ch_scale[label] = 1.0
                elif label.startswith('P'):
                    # assume that the position unit is 'mm'
                    ch_scale[label] = 0.001
                    if ch.GetUnit()=='m': ch_scale[label] = 1.0
            elif fp_type in [2, 3, 4]:
                if label.startswith('F'):
                    # assume that the force unit is 'N'
                    ch_scale[label] = 1.0
                elif label.startswith('M'):
                    # assume taht the torque unit is 'Nmm'
                    ch_scale[label] = 0.001
                    if ch.GetUnit()=='Nm': ch_scale[label] = 1.0
            # assign channel values
            lp_fc = filt_fcs[ch_idx]
            if lp_fc is None:
                ch_data[label] = np.squeeze(ch.GetData().GetValues())
            else:
                ch_data[label] = filt_bw_lp(np.squeeze(ch.GetData().GetValues()), lp_fc, analog_fps, order=filt_order)
        if fp_type == 1:
            cop_l_x_in = ch_data['PX']*ch_scale['PX']
            cop_l_y_in = ch_data['PY']*ch_scale['PY']
            t_z_in = ch_data['TZ']*ch_scale['TZ']
            fx = ch_data['FX']*ch_scale['FX']
            fy = ch_data['FY']*ch_scale['FY']
            fz = ch_data['FZ']*ch_scale['FZ']
            mx = (cop_l_y_in-o_y)*fz+o_z*fy
            my = -o_z*fx-(cop_l_x_in-o_x)*fz
            mz = (cop_l_x_in-o_x)*fy-(cop_l_y_in-o_y)*fx+t_z_in
            f_raw = np.stack([fx, fy, fz], axis=1)
            m_raw = np.stack([mx, my, mz], axis=1)
        elif fp_type == 2:
            f_raw = np.stack([ch_data['FX']*ch_scale['FX'], ch_data['FY']*ch_scale['FY'], ch_data['FZ']*ch_scale['FZ']], axis=1)
            m_raw = np.stack([ch_data['MX']*ch_scale['MX'], ch_data['MY']*ch_scale['MY'], ch_data['MZ']*ch_scale['MZ']], axis=1)
        elif fp_type == 4:
            fp_cal_mat = fp.GetCalMatrix()
            fm_local = np.stack([ch_data['FX'], ch_data['FY'], ch_data['FZ'], ch_data['MX'], ch_data['MY'], ch_data['MZ']], axis=1)
            fm_calib = np.dot(fp_cal_mat, fm_local.T).T
            f_raw = np.stack([fm_calib[:,0]*ch_scale['FX'], fm_calib[:,1]*ch_scale['FY'], fm_calib[:,2]*ch_scale['FZ']], axis=1)
            m_raw = np.stack([fm_calib[:,3]*ch_scale['MX'], fm_calib[:,4]*ch_scale['MY'], fm_calib[:,5]*ch_scale['MZ']], axis=1)
        elif fp_type == 3:
            fx12 = ch_data['FX12']*ch_scale['FX12']
            fx34 = ch_data['FX34']*ch_scale['FX34']
            fy14 = ch_data['FY14']*ch_scale['FY14']
            fy23 = ch_data['FY23']*ch_scale['FY23']
            fz1 = ch_data['FZ1']*ch_scale['FZ1']
            fz2 = ch_data['FZ2']*ch_scale['FZ2']
            fz3 = ch_data['FZ3']*ch_scale['FZ3']
            fz4 = ch_data['FZ4']*ch_scale['FZ4']
            fx = fx12+fx34
            fy = fy14+fy23
            fz = fz1+fz2+fz3+fz4
            mx = fp_len_b*(fz1+fz2-fz3-fz4)
            my = fp_len_a*(-fz1+fz2+fz3-fz4)
            mz = fp_len_b*(-fx12+fx34)+fp_len_a*(fy14-fy23)
            f_raw = np.stack([fx, fy, fz], axis=1)
            m_raw = np.stack([mx, my, mz], axis=1)
        zero_vals = np.zeros((f_raw.shape[0]), dtype=np.float32)
        fm_skip_mask = np.abs(f_raw[:,2])<=threshold
        f_sensor_local = f_raw.copy()
        m_sensor_local = m_raw.copy()
        # filter local values by threshold
        f_sensor_local[fm_skip_mask,:] = 0.0
        m_sensor_local[fm_skip_mask,:] = 0.0
        f_x = f_sensor_local[:,0]
        f_y = f_sensor_local[:,1]
        f_z = f_sensor_local[:,2]
        m_x = m_sensor_local[:,0]
        m_y = m_sensor_local[:,1]
        m_z = m_sensor_local[:,2]
        with np.errstate(invalid='ignore'):
            f_z_adj = np.where(fm_skip_mask, np.inf, f_z)          
            cop_l_x = np.where(fm_skip_mask, np.nan, np.clip((-m_y+(-o_z)*f_x)/f_z_adj+o_x, -fp_len_x*0.5, fp_len_x*0.5))
            cop_l_y = np.where(fm_skip_mask, np.nan, np.clip((m_x+(-o_z)*f_y)/f_z_adj+o_y, -fp_len_y*0.5, fp_len_y*0.5))
            cop_l_z = np.where(fm_skip_mask, np.nan, zero_vals)
            if cop_nan_to_num:
                cop_l_x = np.nan_to_num(cop_l_x)
                cop_l_y = np.nan_to_num(cop_l_y)
                cop_l_z = np.nan_to_num(cop_l_z)
        t_z = m_z-(cop_l_x-o_x)*f_y+(cop_l_y-o_y)*f_x
        # values for the force plate local output
        m_cop_local = np.stack([zero_vals, zero_vals, t_z], axis=1)
        cop_surf_local = np.stack([cop_l_x, cop_l_y, cop_l_z], axis=1)
        f_surf_local = f_sensor_local
        m_surf_local = np.cross(np.array([o_x, o_y, o_z], dtype=np.float32), f_sensor_local)+m_sensor_local
        # values for the force plate global output
        m_cop_global = np.dot(fp_rot_mat, m_cop_local.T).T
        cop_surf_global = np.dot(fp_rot_mat, cop_surf_local.T).T
        f_surf_global = np.dot(fp_rot_mat, f_surf_local.T).T
        m_surf_global = np.dot(fp_rot_mat, m_surf_local.T).T
        # values for the lab output
        m_cop_lab = m_cop_global
        cop_lab = fp_cen+cop_surf_global
        f_cop_lab = f_surf_global
        # prepare return values        
        fp_data.update({'F_SURF_LOCAL': f_surf_local})
        fp_data.update({'M_SURF_LOCAL': m_surf_local})
        fp_data.update({'COP_SURF_LOCAL': cop_surf_local})
        fp_data.update({'F_SURF_GLOBAL': f_surf_global})
        fp_data.update({'M_SURF_GLOBAL': m_surf_global})
        fp_data.update({'COP_SURF_GLOBAL': cop_surf_global})
        fp_data.update({'F_COP_LAB': f_cop_lab})            
        fp_data.update({'M_COP_LAB': m_cop_lab})
        fp_data.update({'COP_LAB': cop_lab})
        if fp_type == 1:
            fp_data.update({'COP_LOCAL_INPUT': np.stack([cop_l_x_in, cop_l_y_in, zero_vals], axis=1)})
        fp_output.update({fp_idx: fp_data})
        fp_idx += 1
    return fp_output

def get_fp_wrench(acq, threshold=0.0):
    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(acq)
    pfe.Update()
    pfc = pfe.GetOutput()
    grwf = btk.btkGroundReactionWrenchFilter()
    grwf.SetInput(pfc)
    grwf.SetTransformToGlobalFrame(True)
    grwf.SetThresholdState(True)
    grwf.SetThresholdValue(threshold)
    grwf.Update()
    grwc = grwf.GetOutput()
    grwc.Update()
    wc_output = {}
    for i in range(grwc.GetItemNumber()):
        wc_data = {}
        pos = grwc.GetItem(i).GetPosition().GetValues()
        force = grwc.GetItem(i).GetForce().GetValues()
        moment = grwc.GetItem(i).GetMoment().GetValues()
        wc_data.update({'POS': pos})
        wc_data.update({'FORCE': force})
        wc_data.update({'MOMENT': moment})
        wc_output.update({i: wc_data})
    return wc_output

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

def get_metadata(acq, parent_name, name):
    md = acq.GetMetaData()
    return md.FindChild(parent_name).value().FindChild(name).value().GetInfo()

def change_point_name(acq, old_name, new_name):
    pt = acq.FindPoint(old_name).value()
    pt.SetLabel(new_name)
    