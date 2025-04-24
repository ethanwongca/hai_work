# VTNet Within Tasks
Each directory contains three VTNet implementations, each designed for a different long-term cognitive ability. The directories included are:
- **`VTNet_1000_att`**  
  Contains VTNet models designed for tasks requiring prolonged attention over 1000 sequences.
- **`VTNet_14seconds_att`**  
  Contains VTNet models configured with a 14-second cutoff, optimized for capturing short-term attention.
- **`VTNet_29seconds_att`**  
  Contains VTNet models configured with a 29-second cutoff, suitable for intermediate long-term cognitive evaluations.
- **`VTNet_29seconds_withoutatt`**  
  Contains VTNet models configured with a 29-second cutoff without attention, suitable for intermediate long-term cognitive evaluations.

Each directory has separate VTNet implementations dedicated to modeling specific long-term cognitive abilities.

# VTNet Modifications 
The following parameters need to be modified in VTNet when running a new experiment:

## 1. Base Directory
Replace the BASE_DIR with your own base directory where `utils.py` and `.pickle` files are located:

```python
# Change this path to your specific setup
BASE_DIR = "/ubc/cs/research/ubc_ml/ewong25"
```

See the VTNet recommended structure for guidance on directory organization.

## 2. Attention Layers
The attention mechanism is implemented with these lines:

```python
# Attention layer definition
self.multihead_attn1 = nn.MultiheadAttention(embed_dim=6, num_heads=1)

# Attention application in forward pass
x2, * = self.multihead_attn1(x2, x2, x2, need_weights=False)
```

To create a non-attention VTNet variant, comment out both of these lines.

## 3. Sequence Length Configuration
In the `st_pickle_loader` function, modify the default `max_length` parameter to match your desired sequence type for cyclic splitting:

```python
def st_pickle_loader(input_file_path, max_length=1000):
    """ Processes a raw data item into a scan path image and a time series
        for input into a STNet.
        Args:
            input_file_name (string): the name of the data item to be loaded
            max_length (int): max number of samples to use for a given item.
                If -1, use all samples
        Returns:
            item (numpy.ndarray): the fully processed data item for RNN input
            item_sp (PIL Image):
    """
```

For cyclic splitting, calculate the max_length as `(seconds * sampling_rate) / 4`:
- `max_length=1000` for 1000 sequences
- `max_length=420` for 14-second experiments (14 * 120 / 4)
- `max_length=870` for 29-second experiments (29 * 120 / 4)

## 4. Data Split Path
In the `cross_validate` function, update the path to your generated pickle file:

```python
def cross_validate(
    model_type,
    folds,
    epochs,
    criterion_type,
    optimizer_type,
    confused_path,
    not_confused_path,
    print_every,
    plot_every,
    hidden_size,
    num_layers,
    down_sample_training=False,
    learning_rate=0.0001,
    path_to_data_split=os.path.join(BASE_DIR, "2verbalwm.pickle"),
    # other parameters...
):
```

Replace the `path_to_data_split` parameter with the path to your own .pickle file generated via the 10-fold cross-validation grouped function in the preprocessing code.

## 5. Preprocessed Dataset Folder
Update the paths to match your preprocessed dataset folder structure:

```python
# Check high low
path_to_sp = os.path.join(BASE_DIR, "2msnv_tasks_processed_29", TASK, "high", "images", filename + '.png')
# Check if the file exists, if not, try "low"
if not os.path.exists(path_to_sp):
    path_to_sp = os.path.join(BASE_DIR, "2msnv_tasks_processed_29", TASK, "low", "images", filename + '.png')
```

Replace `"2msnv_tasks_processed_29"` with the name of your preprocessed dataset folder created during preprocessing.

## 6. Task Name Configuration
Set the appropriate task name variable based on the long-term cognitive state being modeled:

```python
TASK = "VerbalWM_label"
```

Change this to match the specific long-term cognitive task you're working with. If you follow the directory structure in the preprocessing code, you may not need to change this name, but be careful with paths if you use custom names.

To Understand VTNet Read: **_A Neural Architecture for Detecting User Confusion in Eye-tracking Data_** and **_Classification of Alzheimers Disease with Deep Learning on Eye-tracking Data_**
