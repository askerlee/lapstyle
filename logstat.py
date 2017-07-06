import re
import os
import sys
import numpy as np

contentimageAll = ( 'megan.png',  'kid5.png',        'goat.png', 'girlmrf.jpg',    'boy.png' )
styleimageAll   = ( 'spring.png', 'smallworldI.jpg', 'muse.png', 'girlsketch.png', 'girl3.jpg' )
#sigs = ("20_2_100.log", "20_2_100_nobp.log")
#contentLayer2contentLossType = [ -1, 1, 2, -1 ]

sigs = ("relu2_2,relu4_2_20_100_nobp.log",)
contentLayer2contentLossType = [ -1, 0, 1, 2 ]

logpath =sys.argv[1]
readlogcount = 0

max_layer_num = 9
init_losses = []
end_losses = []
end_losses_nobp = []

for i, contentFile in enumerate(contentimageAll):
    styleFile = styleimageAll[i]
    contentName = os.path.splitext(contentFile)[0]
    styleName =   os.path.splitext(styleFile)[0]
    for sig in sigs:
        logfilename = "%s/%s_%s%s" %(logpath, contentName, styleName, sig)
        if not os.path.isfile(logfilename):
            print "'%s' doesn't exist!" %(logfilename)
        else:
            if sig[-8:] == 'nobp.log':
                nobp = True
            else:
                nobp = False
                
            LOG = open(logfilename)
            readlogcount += 1
            iter_type = 'none'
            style_loss_sum = 0
            content_loss_sum = 0
            lap_loss_sum = 0
            
            for line in LOG:
                line = line.strip()
                # Iteration 0 / 1000
                result = re.match(r"Iteration (\d+) / (\d+)", line)
                if result:
                    if iter_type == 'init':
                        loss_block = [1.0, lap_loss_sum/total_loss, total_loss/lap_loss_sum, content_loss_sum/lap_loss_sum, style_loss_sum/lap_loss_sum]
                        init_losses.append(loss_block)
                        lap_loss_sum0 = lap_loss_sum
                    style_loss_sum = 0
                    content_loss_sum = 0
                    lap_loss_sum = 0
                    curr_iter = int(result.group(1))
                    total_iter = int(result.group(2))
                    if curr_iter == 0:
                        iter_type = 'init'
                    elif curr_iter == total_iter - 1:
                        iter_type = 'end'
                    else:
                        iter_type = 'mid'
                        
                    continue
                if iter_type == 'init' or iter_type == 'end':
                    # Content 1 loss: 4858483.750000	
                    result = re.match(r"(Content|Style) (\d+) loss: ([0-9.]+)", line)
                    if result:
                        loss = float(result.group(3))
                        loss_type = result.group(1)
                        layer_no = int(result.group(2))
                        if loss_type == 'Content':
                            contentLossType = contentLayer2contentLossType[layer_no]
                            if contentLossType == 1:
                                content_loss_sum += loss
                            if contentLossType == 2:
                                lap_loss_sum += loss
                        else:
                            style_loss_sum += loss
                    else:
                        result = re.match(r"Total loss: ([0-9.]+)", line)
                        if result:
                            total_loss = float(result.group(1))
                            
            if iter_type == 'end':
                loss_block = [lap_loss_sum/lap_loss_sum0, lap_loss_sum/total_loss, total_loss/lap_loss_sum0,
                            content_loss_sum/lap_loss_sum0, style_loss_sum/lap_loss_sum0]
                if nobp:
                    end_losses_nobp.append(loss_block)
                else:
                    end_losses.append(loss_block)
                    
print "%d log files read" %readlogcount
init_losses = np.array(init_losses)
#arr_labels = ('init_losses', 'end_losses', 'end_losses_nobp')
# always init_losses_nobp == init_losses. So no need to print it
#for i,arr in enumerate([init_losses, end_losses, end_losses_nobp]):
arr_labels = ('init_losses', 'end_losses_nobp')
for i,arr in enumerate([init_losses, end_losses_nobp]):
    arr = np.array(arr)
    arr_label = arr_labels[i]
    loss_labels = ('lap', 'lap-frac', 'total', 'content', 'style')
    print "%s:" %arr_label
    for j,loss_label in enumerate(loss_labels):
        losses = arr[:,j]
        mean = np.mean(losses)
        std = np.std(losses)
        print "%s: %.5f (%.5f)" %(loss_label, mean, std)
        print losses
        