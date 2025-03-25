#!/bin/bash

#models=("ywhy" "dhzy" "gczy" "qzzy")
models=("scxc")

total=${#models[@]}
# utils defination
get_timestamp(){
	date +"%Y-%m-%d %H:%M:%S"
}

get_libsuffix(){
	date +"%Y%m%d"
}

show_progress(){
	local current=$1
	local total=$2
	local percent=$((current * 100 / total))
	# 50 "=" in total, 10 "=" for every model
	local progress=$((current * 50 / total))
	# printf "\r\n["
	printf "["
	for ((i=0; i<50; i++)); do
		if [ $i -lt $progress ]; then
			printf "="
		else
			printf " "
		fi
	done
	printf "] %d%%\r\n" "$percent"
}

# set env
if [ $# -ne 1 ]; then
	echo "[error] $(get_timestamp), invalid params, requires 1 param"
	echo "[info] $(get_timestamp), Usage: $0 <path/to/model>. path requires .name and .engine files"
	exit 0
else
	model_path=$1
	cd "$model_path" || exit 1
fi

# generate so file
counter=1
for model in "${models[@]}";do
	# generate model_names.h
	echo "begin to generate ${model}_names.h... "
	if [ -e "${model}.names" ]; then
		echo "const char * ${model}_names[]={" > "${model}_names.h"
		cat "${model}.names" | sed 's/^/"/; s/$/",/' >> "${model}_names.h" 
		echo "};" >> "${model}_names.h"
		echo "unsigned int class_num = sizeof(${model}_names) / sizeof(${model}_names[0]);" >> "${model}_names.h"
		echo "[info] $(get_timestamp), ${model}_names.h generate success"
	else
		echo "[error] $(get_timestamp), no ${model}.names exits, skip to next model"
		continue
	fi

	echo "begin to generate ${model}.h... "
	# generate model.h
	if [ -e "${model}.engine" ]; then	
		xxd -i "${model}.engine" > "${model}.h"
		echo "[info] $(get_timestamp), ${model}.h generate success"
	else
		echo "[error] $(get_timestamp), no ${model}.engine exits, skip to next model"
		continue
	fi

	echo "begin to generate ${model}.cpp... "
	# generate model.cpp
	cat << EOF > "${model}.cpp"
#include "${model}.h"
#include "${model}_names.h"
#include <iostream>
#include <string.h>

bool IsAuthorized(const char * value)
{
    const char *key = "jiayang2024";
    return strcmp(key, value) == 0;
}

extern "C"
{
    unsigned char *getModelData(const char * key)
    {
        if(IsAuthorized(key))
        {
            return ${model}_engine;
        }
        return nullptr;
    }

    unsigned int getModelLen()
    {
        return ${model}_engine_len;
    }

    unsigned int getClassNum()
    {
        return class_num;
    }

    const char **getClassName()
    {
        return ${model}_names;
    }
}
EOF
	echo "[info] $(get_timestamp), ${model}.cpp generate success"
	echo "[info] $(get_timestamp), ${model} head and cpp files generate, begin to compile library.\ 
	Please wait patiently(approximately 5mins, yolov11x maybe needs 10mins or more)..."

	# generate model library
	 g++ "${model}.cpp" -fPIC -shared -O2 -o "lib_nvidia_${model}.so.1.1.$(get_libsuffix)"
	if [ -e "lib_nvidia_${model}.so.1.1.$(get_libsuffix)" ]; then
		echo "[info] $(get_timestamp), lib_nvidia_${model}.so.1.1.$(get_libsuffix) generate success"
		rm ${model}.h ${model}_names.h ${model}.cpp
	else
		echo "[error] $(get_timestamp), lib_nvidia_${model}.so.1.1.$(get_libsuffix) generate fail"
	fi
	show_progress $((counter++)) $total
	
done

printf "\n"
