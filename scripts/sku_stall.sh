stall=$1

awk -F'/' '{
    if(a[$1]=="") a[$1]=$2; 
    else a[$1]=a[$1]"，"$2
} 
END {
    for(i in a) print i "：\n" a[i] "\n"
}' ${stall}
