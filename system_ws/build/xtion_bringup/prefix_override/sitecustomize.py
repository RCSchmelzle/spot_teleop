import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rysch01/Projects/teleoperation_spot/system_ws/install/xtion_bringup'
