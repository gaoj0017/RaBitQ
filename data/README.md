
# Prerequisites
*   Python == 3.8, numpy == 1.20.3, faiss == 1.7.4, tqdm

# Datasets

The tested datasets are available at https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html. 

## Reproduction
1. Download a dataset. The data format of ``.fvecs'' can be found in http://corpus-texmex.irisa.fr/.
    
2. Unzip the dataset. 

3. Build the IVF index for the dataset. 

    ```shell
    python ivf.py
    ```

4. Run the index phase of RaBitQ for the dataset. 

    ```shell
    python rabitq.py
    ```