# Iterative Optimization Code

## __How to run__

1. To run with the provided data [download](https://drive.google.com/drive/folders/1CaGBEdLhVGDygo5mfhdSRltkxHBBN6K9?usp=sharing) the:
    - embeddings 
    - quality scores
    - clustering information

    If you wish to run with custom data you will have to generate these files yourself and save them as pickled files:

    - embeddings in the form {"image_name": embedding (np.array, dtype=float32)}
    - quality scores using some FIQA in the form {"image_name": quality_score (float)}
    - clustering information {"identity": {"cluster_id": ["cluster_samples"]}}

> In the file "clustering_helper.py" you can find the code for generating the clustering data!

2. Store these files in "./base_data"

3. Then run the main script __optimize.py__
    - for the provided data run this command:
        
        _python3 optimize.py -ql ./base_data/faceqan-vggface2-quality.pkl -sl ./improved-quality.pkl -el ./base_data/arcface-vggface2-embeddings.pkl -cl ./base_data/vggface2-clustering.pkl_



