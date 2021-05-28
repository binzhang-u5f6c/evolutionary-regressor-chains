#!/bin/bash

rsync -r config bzhang3@$1.eng.uts.edu.au:/home/bzhang3/Running/Code.0$2/
rsync -r erc bzhang3@$1.eng.uts.edu.au:/home/bzhang3/Running/Code.0$2/
rsync *.py bzhang3@$1.eng.uts.edu.au:/home/bzhang3/Running/Code.0$2/
