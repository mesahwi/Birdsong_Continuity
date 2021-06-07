#!/bin/sh
mkdir KNN_results
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TbuQttfgIiSXHe6i0ihyl3N4cesnH1Eo' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TbuQttfgIiSXHe6i0ihyl3N4cesnH1Eo" -O Data.zip && rm -rf ~/cookies.txt
unzip Data.zip
