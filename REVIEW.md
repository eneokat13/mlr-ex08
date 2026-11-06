1.

<img width="585" height="1266" alt="IMG_1693" src="https://github.com/user-attachments/assets/d68f2c61-3dde-474c-ac59-146d6f53581b" />
<img width="585" height="1266" alt="IMG_1692" src="https://github.com/user-attachments/assets/fb9e7ed5-53a0-41aa-8e74-d7ff5d0497c3" />

Here are images showing that the URL Page opens and works on phones, classifying images in the live camera and showing the probabilities correctly.

2.

<img width="702" height="855" alt="2026-11-06 16_46_43-NVIDIA GeForce Overlay" src="https://github.com/user-attachments/assets/a4fd03c3-5aab-404b-aafa-acc3707d05c7" />

Uploading check_image.jpg, we can see that the model correctly predicts the leaf, having the same probabilites when compared to check_expected.json

3.

Where the frame is cropped:

In line 543 you can find the following code that successfully crops the frame,

var side = Math.min(srcW, srcH);
var sx = (srcW - side) / 2;
var sy = (srcH - side) / 2;


Where the frame is resized to 96 x 96:

In line 542 "var S = state.info.input_size;" gets from our model_info.json file the value of S, for our case the number 96,

And in code line 556 we then draw the image with respect to the value S,
workCtx.drawImage(
    srcEl,
    sx, sy, side, side,
    0, 0, S, S
);


Where is it divied by 255:

In line 564 we can find the following,
rgba[o] / 255
rgba[o + 1] / 255
rgba[o + 2] / 255


Where it is standardized with mean and std:

In the same lines the data is standardized,
data[i]         = (rgba[o]     / 255 - mean[0]) / std[0];
data[n + i]     = (rgba[o + 1] / 255 - mean[1]) / std[1];
data[2 * n + i] = (rgba[o + 2] / 255 - mean[2]) / std[2];


In which order the channels go into the tensor:

The channels are stored in this order:
data[i] - Red (R)
data[n+1] - Green (G)
Data[2*n+1] - Blue (B)

And in code line 572 "return new ort.Tensor('float32', data, [1, 3, S, S]);",
We can see that the final shape is [1, 3, 96, 96],
1 is the batch size
3 are the RGB channels
96 image width and height


Does it match model_info.json and the transform of the notebook:

The division by 255, mean, standard deviation, RGB channel order, and tensor layout are consistent with the notebook's ToTensor() and Normalize() operations.

But,
The website performs center cropping and resizing, but the notebook transform you showed does not.

