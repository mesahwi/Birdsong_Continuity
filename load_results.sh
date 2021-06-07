#!/bin/sh
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fyxnsQKH34uyvpr89fC3CG6ac1mT4fKg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fyxnsQKH34uyvpr89fC3CG6ac1mT4fKg" -O KNN_results.zip && rm -rf ~/cookies.txt
mkdir KNN_results
unzip KNN_results.zip -d KNN_results