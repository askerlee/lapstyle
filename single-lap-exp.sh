#!/bin/bash -x
# test lap_style.lua with default settings on five test image pairs.
# You can specify a different laplacian layer(s) using laplayersAll and lapweightsAll.

contentimageAll=( megan.png  kid5.png        goat.png girlmrf.jpg    boy.png)
styleimageAll=(   spring.png smallworldI.jpg muse.png girlsketch.png girl3.jpg)
inputLen=${#contentimageAll[@]}
contentweight=20
#laplayersAll=( 2   2,4)
#lapweightsAll=(100 100,1600)
laplayersAll=( 2)
lapweightsAll=(100)
configLen=${#laplayersAll[@]}
for (( i=0; i<$inputLen; i++ )); do
    contentimage=${contentimageAll[$i]}
    contentname="${contentimage%.*}"
    styleimage=${styleimageAll[$i]}
    stylename="${styleimage%.*}"
    for (( j=0; j<$configLen; j++ )); do
        laplayers=${laplayersAll[$j]}
        lapweights=${lapweightsAll[$j]}
        outsig="${contentname}_${stylename}${contentweight}_${laplayers}_${lapweights}"
        th lap_style.lua -style_image $styleimage -content_image $contentimage -output_image ${outsig}.png -content_weight $contentweight -lap_layers $laplayers -lap_weights $lapweights 2>&1| tee ${outsig}.log
        outsig="${contentname}_${stylename}${contentweight}_${laplayers}_${lapweights}_nobp"
        th lap_style.lua -style_image $styleimage -content_image $contentimage -output_image ${outsig}.png -content_weight $contentweight -lap_layers $laplayers -lap_weights $lapweights -lap_nobp 2>&1| tee ${outsig}.log    
    done
done
