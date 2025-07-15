Getting started:

pip install -r requirements.txt

Add your raw data files to the data/RAW_DATA folder.

python main.py

After running, the output .pkl and .csv files will be saved in the output/DAILY_SUMMARIES folder.

python plotting/plot_quantile_bars.py

After running, the pdf of visuals will be saved in the output folder and each image as a png will be saved in the outputs/PLOTS folder.


NEED TO CHANGE: 
- get rid of nested loops to make the code more efficient for larger input data sets.
   - possibly use data structures in Python for example.
- need to verify calculations are correct for various statistics.
