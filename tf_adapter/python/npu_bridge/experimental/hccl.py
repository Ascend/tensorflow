import os
import ctypes

def load_lib(lib_name):
    try:
        lib = ctypes.CDLL(lib_name)
    except Exception as e:
        raise ValueError('load lib ', lib_name, ' error')

    return lib

hccl_graph_adp_ctypes = load_lib('libhcom_graph_adaptor.so')

def c_str(string):
    return ctypes.c_char_p(string.encode('utf-8'))

def get_actual_rank_size(group="hccl_world_group"):
    c_group = c_str(group)
    c_rank_size = ctypes.c_uint()
    ret = hccl_graph_adp_ctypes.HcomGetActualRankSize(c_group, ctypes.byref(c_rank_size))
    if ret != 0:
        raise ValueError('get actual rank size error.')
    return c_rank_size.value

def get_user_rank_size():
    rank_size = int(os.getenv('RANK_SIZE'))
    return rank_size

def get_user_rank_id():
    rank_id = int(os.getenv('RANK_ID'))
    return rank_id