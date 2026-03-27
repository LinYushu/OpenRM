#!/bin/bash

blue="\033[1;34m"
yellow="\033[1;33m"
red="\033[1;31m"
reset="\033[0m"

include_count=$(find include -type f \( -name "*.h"  -o -name "*.cuh" \) -exec cat {} \; 2>/dev/null | wc -l)
src_count=$(find src -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.txt" \) -exec cat {} \; 2>/dev/null | wc -l)
cuda_count=$(find cuda/src cuda/include -type f \( -name "*.cuh" -o -name "*.cu" -o -name "*.h" -o -name "*.txt" \) -exec cat {} \; 2>/dev/null | wc -l)
terminal_count=$(find terminal/src terminal/include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.txt" \) -exec cat {} \; 2>/dev/null | wc -l)
daheng_count=$(find include/video/daheng -type f \( -name "*.h" \) -exec cat {} \; 2>/dev/null | wc -l)

total=$((include_count + src_count + cuda_count + terminal_count - daheng_count))

max_threads=$(nproc 2>/dev/null || echo 4)

if [ ! -d "build" ]; then 
    mkdir build
fi

setup_ldconfig() {
    if [ ! -f "/etc/ld.so.conf.d/usr-local-lib.conf" ]; then
        echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/usr-local-lib.conf > /dev/null
    fi
    sudo ldconfig
}

skip_build=0

while getopts ":ritdg:k" opt; do
    case $opt in
        r)
            echo -e "${yellow}<<<--- rebuild --->>>\n${reset}"
            cd build
            sudo make uninstall
            cd ..
            sudo rm -rf build
            mkdir build
            ;;
        i)
            echo -e "${yellow}<<<--- reinstall --->>>\n${reset}"
            cd build
            sudo make uninstall
            cd ..
            ;;
        t)
            echo -e "${yellow}<<<--- terminal --->>>\n${reset}"
            cd build
            if [ $skip_build -eq 0 ]; then
                cmake ..
                make -j "$max_threads"
            else
                echo -e "${yellow}\n<<<--- skip cmake & make --->>>\n${reset}"
            fi
            sudo make install
            cd ..
            setup_ldconfig
            cd terminal
            if [ ! -d "build" ]; then 
                mkdir build
            fi
            cd build
            if [ $skip_build -eq 0 ]; then
                cmake ..
                make -j "$max_threads"
            else
                echo -e "${yellow}\n<<<--- skip terminal cmake & make --->>>\n${reset}"
            fi
            sudo make install
            cd ../..
            sudo ldconfig
            exit 0
            ;;
        d)
            echo -e "${yellow}\nThe following files and directories will be deleted:${reset}"
            sudo find "$(pwd)" -maxdepth 1 -name "build"
            sudo find /usr/local/ -name "*openrm*"
            
            echo -e "${red}Are you sure you want to delete openrm? (y/n): ${reset}"
            read answer
            if [[ "$answer" =~ ^[Yy]$ ]]; then
                echo -e "${yellow}\n<<<--- delete --->>>\n${reset}"
                sudo find "$(pwd)" -maxdepth 1 -name "build" -exec rm -rf {} +
                sudo find /usr/local/ -name "*openrm*" -exec rm -rf {} +
            else
                echo -e "${yellow}Deletion operation canceled${reset}"
            fi
            exit 0
            ;;
        g)
            git_message=$OPTARG
            echo -e "${yellow}\n<--- Git $git_message --->${reset}"
            git pull
            git add -A
            git commit -m "$git_message"
            git push
            exit 0
            ;;
        k)
            skip_build=1
            ;;
        \?)
            echo -e "${red}\n--- Unavailable param: -$OPTARG ---\n${reset}"
            ;;
        :)
            echo -e "${red}\n--- param -$OPTARG need a value ---\n${reset}"
            ;;
    esac
done

cd build

if [ $skip_build -eq 0 ]; then
    echo -e "${yellow}\n<<<--- start cmake --->>>\n${reset}"
    cmake ..

    echo -e "${yellow}\n<<<--- start make --->>>\n${reset}"
    make -j "$max_threads"
else
    echo -e "${yellow}\n<<<--- skip cmake & make (installing directly) --->>>\n${reset}"
fi

echo -e "${yellow}\n<<<--- start install --->>>\n${reset}"
sudo make install
setup_ldconfig

echo -e "${yellow}\n<<<--- Total Lines --->>>${reset}"
echo -e "${blue}           $total${reset}"

echo -e "${yellow}\n<<<--- Welcome OpenRM --->>>\n${reset}"