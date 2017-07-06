#!/bin/bash -x
# test lap_style.lua with the combination of multiple laplacian layers, on a single image pair

contentimg=$1
styleimg=$2
contentweight=$3
contentFilename=$(basename "$contentimg")
contentname="${contentFilename%.*}"
styleFilename=$(basename "$styleimg")
stylename="${styleFilename%.*}"
laplayersAll=( 1  1   2    1,2     1,3     2,3      2,4)
lapweightsAll=(0  50  100  50,100  50,400  100,400  100,1600)
configLen=${#laplayersAll[@]}
for (( i=0; i<$configLen; i++ )); do
    laplayers=${laplayersAll[$i]}
    lapweights=${lapweightsAll[$i]}
    outsig="${contentname}_${stylename}${contentweight}_${laplayers}_${lapweights}"
    th lap_style.lua  -style_image images/$styleimg -content_image images/$contentimg -output_image output/${outsig}.png -content_weight $contentweight -lap_layers $laplayers -lap_weights $lapweights 2>&1| tee output/${outsig}.log
done
