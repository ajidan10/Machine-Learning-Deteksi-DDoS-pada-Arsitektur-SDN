#!/bin/bash
n=3     # number of switches
for i in {1..2000}
do
    echo "Pemeriksaan jaringan... "
    for ((j = 1; j <= n; j++))
    do
        
        
        # extract essential data from raw data
        sudo ovs-ofctl dump-flows s$j > data/raw
        
        grep "nw_src" data/raw > data/flowentries.csv
	
	# Ekstrak informasi forward dan backward
	all_info=$(cat data/raw)
	forward_info=$(echo "$all_info" | grep "nw_src=10.0.0.5" )
	backward_info=$(echo "$all_info" | grep "nw_dst=10.0.0.5")
	echo "$forward_info" > data/forward_info.csv
	echo "$backward_info" > data/backward_info.csv
	
	pkt=$(awk -F "," '{split($9,a,"="); print a[2]","}' data/flowentries.csv)
	packets=$(awk -F "," '{split($4,a,"="); print a[2]","}' data/flowentries.csv)
        bytes=$(awk -F "," '{split($5,b,"="); print b[2]","}' data/flowentries.csv)
        fwd_pkt=$(awk -F "," '{split($9,a,"="); print a[2]","}' data/forward_info.csv)
        fwd_packets=$(awk -F "," '{split($4,a,"="); print a[2]","}' data/forward_info.csv)
        fwd_bytes=$(awk -F "," '{split($5,b,"="); print b[2]","}' data/forward_info.csv)
        bwd_pkt=$(awk -F "," '{split($9,a,"="); print a[2]","}' data/backward_info.csv)
        bwd_packets=$(awk -F "," '{split($4,a,"="); print a[2]","}' data/backward_info.csv)
        bwd_bytes=$(awk -F "," '{split($5,b,"="); print b[2]","}' data/backward_info.csv)
        ipsrc=$(awk -F "," '{out=""; for(k=2;k<=NF;k++){out=out" "$k}; print out}' data/flowentries.csv | awk -F " " '{split($11,d,"="); print d[2]","}')
        ipdst=$(awk -F "," '{out=""; for(k=2;k<=NF;k++){out=out" "$k}; print out}' data/flowentries.csv | awk -F " " '{split($12,d,"="); print d[2]","}')
       
       
        # check if there are no traffics in the network at the moment.
        if test -z "$fwd_packets" || test -z "$fwd_bytes" || test -z "$bwd_packets" || test -z "$bwd_bytes" || test -z "$packets" || test -z "$bytes" || test -z "$ipsrc" || test -z "$ipdst"
        then
            state=0
        else
            echo "$pkt" > data/pkt.csv
            echo "$packets" > data/packets.csv
            echo "$bytes" > data/bytes.csv
            echo "$fwd_pkt" > data/fwd_pkt.csv
            echo "$fwd_packets" > data/fwd_packets.csv
            echo "$fwd_bytes" > data/fwd_bytes.csv
            echo "$bwd_pkt" > data/bwd_pkt.csv
            echo "$bwd_packets" > data/bwd_packets.csv
            echo "$bwd_bytes" > data/bwd_bytes.csv
            echo "$ipsrc" > data/ipsrc.csv
            echo "$ipdst" > data/ipdst.csv
            
            for csv_file in data/*.csv; do
                tac "$csv_file" > "$csv_file.temp" && mv "$csv_file.temp" "$csv_file"
            done
             
            
            python3 2.py 
            python3 inspector.py
            state=$(awk '{print $0;}' .result)
        fi
        if [ $state -eq 0 ]; then
            echo "Jaringan Normal pada s$j"
	else 
            echo " PERINGATAN! Terdeteksi Ancaman Atau Serangan Pada Jaringan!"
	

            default_flow=$(sudo ovs-ofctl dump-flows s$j | tail -n 1)    # Get flow "action:CONTROLLER:<port_num>" sending unknown packet to the controller
            sudo ovs-ofctl del-flows s$j
            sudo ovs-ofctl add-flow s$j "$default_flow"
        fi
        
    done
    sleep 3
done

