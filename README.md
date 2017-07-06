# Laplacian-Steered Neural Style Transfer
Code and test images for paper "[Laplacian-Steered Neural Style Transfer](https://arxiv.org/abs/1707.01253)".

The Laplacian loss has been extended on the following three neural style transfer implementations: 

* **lap_style.lua** - https://github.com/jcjohnson/neural-style Gatys-style[1] implemented by Justin Johnson, using the L-BFGS optimization method.
* **tf-neural-style/neural_style.py** - https://github.com/anishathalye/neural-style Gatys-style[1] by Anish Athalye, using Adam. 
* **neural-doodle/doodle.py** - https://github.com/alexjc/neural-doodle MRF-CNN[2] implemented by Alex J. Champandard.

The implementation by Justin Johnson clearly produces the best images. It seems the L-BFGS optimization is the only reason, because this algorithm is otherwise identical to Anish Athalye's implementation.

Sample usage:
```
th lap_style.lua -style_image images/flowers.png -content_image images/megan.png -output_image output/megan_flowers20_100.png -content_weight 20 -lap_layers 2 -lap_weights 100
```

Sample images:

Note: although photo-realistic style transfer[3] (https://github.com/luanfujun/deep-photo-styletransfer) performs amazingly well on their test images, it doesn't work on the images we tested. Seems that in order to make it work well, the content image and the style image has to have highly similar layout and semantic contents.

You are welcome to cite the paper (https://arxiv.org/abs/1707.01253) with this bibtex:

```
@InProceedings{lapstyle,
  author    = {Shaohua Li and Xinxing Xu and Liqiang Nie and Tat-Seng Chua},
  title     = {Laplacian-Steered Neural Style Transfer},
  booktitle = {Proceedings of the ACM Multimedia Conference (MM), to appear.},
  year      = {2017},
}
```

### References
[1] Leon A Gatys, Alexander S Ecker,and Matthias Bethge. 2016. Image style transfer using convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2414–2423.

[2] Chuan Li and Michael Wand. 2016. Combining markov random fields and convolutional neural networks for image synthesis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2479–2486.

[3] Fujun Luan, Sylvain Paris, Eli Shechtman, and Kavita Bala. 2017. Deep Photo Style Transfer. arXiv preprint arXiv:1703.07511 (2017).
