What each change brought:

From the task4_log.txt, we have the following table,

| Experiment | Epochs | Val accuracy | Camera accuracy | Best epoch |

| Baseline (Task 3, no augmentation) | 15 | 80.8% | 60.0% | 10 |
| + geometric augmentation | 15 | 86.7% | 62.5% | 13 |
| + colour jitter (on top of geometric augmentation) | 15 | 73.3% | 69.2% | 13 |
| + global average pooling | 15 | 74.2% | 55.0% | 12 |
| + dropout 0.25 | 15 | 82.5% | 60.0% | 13 |
| + AdamW, weight decay 1e-4 | 15 | 80.8% | 60.0% | 8 |
| + label smoothing 0.1 | 15 | 80.8% | 62.5% | 6 |

| MobileNetV3 Small, frozen backbone (train head only) | 15 | 83.3% | 69.2% | 10 |
| MobileNetV3 Small, fine-tune whole network | 15 | 91.7% | 75.8% | 2 |
| MobileNetV3 Small + augmentation + colour jitter + cosine schedule | 30 | 88.3% | 91.7% | 8 |

| Scratch combination: augmentation + colour + GAP + dropout + AdamW + label smoothing | 30 | 63.3% | 62.5% | 24 |
| Transfer combination + best-epoch checkpoint | 30 | 88.3% | 88.3% | 3 |

We can see how the baseline from task 3 WITH geometric augmentation improved its accuracy,
Colour jitter (on top of geometric augmentation) also returned better camera results


What did not help:

We can see how global average pooling did not help and camera accuracy fell to 55%
Dropout 0.25 returned small Val accuracy gain, while camera accuracy remained the same
AdamW did not improve either accuracy
Label smoothing kept Val accuracy the same, and increased Camera accuracy slightly


Was the difference larger than the noise:

Yes, for the main improvement, the difference was larger than the observed noise.
The baseline varied by only 1.7 percentage points between the two runs.


Which model would you put on the phone:

The model with the best validation accuracy (88.3%) and camera accuracy (91.7%), so i would choose that model.
