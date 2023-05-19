# Transfer Learning Code

> You can use your own training scripts on the optimized labels or run the provided script as described below. 


## __How to run__

1. The transfer learning relies on the pretrained ArcFace model, the weights can be obtained from the [official Github repository](https://github.com/deepinsight/insightface)

    - Download the weights and place them in "./base_model", rename the file to "weights.pth"

2. To properly run the transfer learning you also need quality scores (base or optimized)


> Naturally you can also use other pretrained models, by placing the source code and weights in "./base_model" and adjusting the main script to load your desired model.


3. Then run the main scrip __train_base.py__

    - for the provided data run this command:
        
        _python3 train_base.py -sl ./model.pth -ql ./../iterative_optimization/improved-quality.pkl_




