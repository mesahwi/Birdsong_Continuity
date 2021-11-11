#!/bin/sh
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18YIPI5B-tQ5_Ded82eVBuSKZe8xmpnN3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18YIPI5B-tQ5_Ded82eVBuSKZe8xmpnN3" -O KNN_results.zip && rm -rf ~/cookies.txt
mkdir KNN_results
unzip KNN_results.zip -d KNN_results
