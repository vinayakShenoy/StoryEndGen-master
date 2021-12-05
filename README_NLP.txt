
Instructions to generate an inference file:

1. Clone the git respository at master.
2. Create a train folder in the source directory.
3. Copy the checkpoint files from the google drive link to the train folder. Each folder in the drive has a set of 3 checkpoint files. Copy those to the train folder.
4. Run the inference command as follows:
python main.py --is_train=False --inference_path='xxx' --inference_version='yyy'

xxx: Name of the folder where you want to generate and keep the inference files
yyy: Integer value after the checkpoint file name. You can create different inference files based on the checkpoint file number.

