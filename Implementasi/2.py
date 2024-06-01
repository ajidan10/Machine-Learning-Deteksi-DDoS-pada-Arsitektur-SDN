import numpy as np
import csv

# Menghitung Ukuran Paket Bwd
bwd_pkt_csv = np.genfromtxt('data/bwd_packets.csv', delimiter=",")
pkt_bwd = bwd_pkt_csv[0]
# Tot Bwd Pkts
tot_bwd_pkts = np.nan_to_num(np.sum(pkt_bwd), nan=0)
# Bwd Pkt Len Max
bwd_pkt_len_max = np.nan_to_num(np.max(pkt_bwd), nan=0)
# Bwd Pkt Len Mean
bwd_pkt_len_mean = np.nan_to_num(np.mean(pkt_bwd), nan=0)
# Bwd Pkt Len Std
bwd_pkt_len_std = np.nan_to_num(np.std(pkt_bwd), nan=0)
# Bwd Header Len
bwd_header_len = 40
#Menghitung Ukuran Paket Fwd
fwd_pkt_csv = np.genfromtxt('data/fwd_packets.csv', delimiter=",")
pkt_fwd_csv = fwd_pkt_csv[0]
pkt_fwd = np.nan_to_num(np.sum(pkt_fwd_csv), nan=0)
# Tot Fwd Pkts
tot_fwd_pkts = np.nan_to_num(np.sum(pkt_fwd), nan=0)
# Fwd Pkt Len Max
fwd_pkt_len_max = np.nan_to_num(np.max(pkt_fwd), nan=0)
# Fwd Pkt Len Mean
fwd_pkt_len_mean = np.nan_to_num(np.std(pkt_fwd), nan=0)
# Fwd Header Len
fwd_header_len = 0
# Menghitung Jumlah paket per Interval
#Flow Pkts/s
packets_csv = np.genfromtxt('data/packets.csv', delimiter=",")
if packets_csv.ndim == 1:
    packets = packets_csv[0]
else:
    packets = packets_csv[:, 0]
tot_pkt = np.nan_to_num(np.sum(packets), nan=0)
flow_pkts_per_sec = tot_pkt 
# Fwd Pkts/s
fwd_packets_csv = np.genfromtxt('data/fwd_packets.csv', delimiter=",")
if fwd_packets_csv.ndim == 1:
    packets_fwd = fwd_packets_csv[0]
else:
    packets_fwd = fwd_packets_csv[:,0]
tot_pkt_fwd = np.nan_to_num(np.sum(packets_fwd), nan=0)
fwd_pkts_per_sec = tot_pkt_fwd 
# Bwd Pkts/s
bwd_packets_csv = np.genfromtxt('data/bwd_packets.csv', delimiter=",")
if bwd_packets_csv.ndim == 1:
    packets_bwd = bwd_packets_csv[0]
else:
    packets_bwd = bwd_packets_csv[:,0]
tot_pkt_bwd = np.nan_to_num(np.sum(packets_bwd), nan=0)
bwd_pkts_per_sec = tot_pkt_bwd 
# Menghitung Ukuran Paket
# Pkt Len Max
pkt_max_csv = np.genfromtxt('data/fwd_packets.csv', delimiter=",")
if pkt_max_csv.ndim == 1:
    pkt_max = pkt_max_csv[0]
else:
    pkt_max = pkt_max_csv[:, 0]
pkt_len_max = np.nan_to_num(np.max(pkt_max), nan=0)
# Pkt Len Mean
pkt_mean_csv = np.genfromtxt('data/fwd_packets.csv', delimiter=",")
if pkt_mean_csv.ndim == 1:
    pkt_mean = pkt_mean_csv[0]
else:
    pkt_mean = pkt_mean_csv[:, 0]
pkt_len_mean = np.nan_to_num(np.mean(pkt_mean), nan=0)
# Pkt Len Std
pkt_std_csv = np.genfromtxt('data/fwd_packets.csv', delimiter=",")
if pkt_std_csv.ndim == 1:
    pkt_std = pkt_std_csv[0]
else:
    pkt_std = pkt_std_csv[:, 0]
pkt_len_std = np.nan_to_num(np.std(pkt_std), nan=0)

# FIN Flag Count
fin_flag_cnt = 0
# SYN Flag Count
syn_flag_cnt = 0

# Subflow Fwd Pkts
fwd_subflow_pkt_csv = np.genfromtxt('data/fwd_packets.csv', delimiter=",")
subflow_fwd_pkt = fwd_subflow_pkt_csv[0]
subflow_fwd_pkts = np.nan_to_num(np.sum(subflow_fwd_pkt), nan=0)
# Packet Size Average
bytes_csv = np.genfromtxt('data/fwd_packets.csv', delimiter=",")
if bytes_csv.ndim == 1:
    bytes_size = bytes_csv[0]
else:
    bytes_size = bytes_csv[:, 0]
tot_byte_forwarded = np.nan_to_num(np.sum(bytes_size), nan=0)
pkt_size_avg = tot_byte_forwarded 
# Fwd Seg Size Avg
bytes_fwd_size_csv = np.genfromtxt('data/fwd_bytes.csv', delimiter=",")
if bytes_fwd_size_csv.ndim == 1:
    bytes_fwd_size = bytes_fwd_size_csv[0]
else:
    bytes_fwd_size = bytes_fwd_size_csv[:, 0]
tot_byte_forwarded_fwd = np.nan_to_num(np.sum(bytes_fwd_size), nan=0)
fwd_seg_size_avg = tot_byte_forwarded_fwd 
# Subflow Bwd Pkts
bwd_subflow_pkt_csv = np.genfromtxt('data/bwd_packets.csv', delimiter=",")
subflow_bwd_pkt = fwd_subflow_pkt_csv[0]
subflow_bwd_pkts = np.nan_to_num(np.sum(subflow_bwd_pkt), nan=0)
# Subflow Bwd Byts
bwd_subflow_byts_csv = np.genfromtxt('data/bwd_bytes.csv', delimiter=",")
subflow_bwd_byts = bwd_subflow_byts_csv[0]
subflow_bwd_byts = np.nan_to_num(np.sum(subflow_bwd_byts), nan=0)
# Init Bwd Win Byts

bwd_bytes_csv = np.genfromtxt('data/bwd_pkt.csv', delimiter=",")
if bwd_bytes_csv.ndim == 1:
    bytes_bwd = bwd_bytes_csv[0]
else:
    bytes_bwd = bwd_bytes_csv[:,0]
max_bytes_bwd = np.nan_to_num(np.max(bytes_bwd), nan=0) 
min_bytes_bwd = np.nan_to_num(np.mean(bytes_bwd), nan=0)
init_bwd_win_byts = (min_bytes_bwd) - (max_bytes_bwd)

# Membuat Header
header = [
    "Tot Fwd Pkts", "Tot Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Mean", 
    "Bwd Pkt Len Max", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Pkts/s", 
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Max",
    "Pkt Len Mean", "Pkt Len Std", "FIN Flag Cnt", "SYN Flag Cnt", "Pkt Size Avg", 
    "Fwd Seg Size Avg", "Subflow Fwd Pkts","Subflow Bwd Pkts", "Subflow Bwd Byts", 
    "Init Bwd Win Byts"
]
# Membuat Training Data
smart = [
    tot_fwd_pkts, tot_bwd_pkts, fwd_pkt_len_max, fwd_pkt_len_mean, bwd_pkt_len_max, 
    bwd_pkt_len_mean, bwd_pkt_len_std, flow_pkts_per_sec, fwd_header_len, bwd_header_len,
    fwd_pkts_per_sec, bwd_pkts_per_sec, pkt_len_max, pkt_len_mean, pkt_len_std, 
    fin_flag_cnt, syn_flag_cnt, pkt_size_avg, fwd_seg_size_avg, subflow_fwd_pkts, 
    subflow_bwd_pkts, subflow_bwd_byts, init_bwd_win_byts
]
# masukan ke dalam file CSV
with open('realtime.csv', 'w') as datafile:
    writer = csv.writer(datafile, delimiter=",")
    writer.writerow(header)
    writer.writerow(smart)
    
    datafile.close()
