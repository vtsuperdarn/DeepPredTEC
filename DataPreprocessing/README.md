# Input data pipeline
These files are used for pre-processing and storing the TEC maps in 2D numpy arrays. 

## File structure and details:
* `generate_tec_map_files.py`: This file reads in the median filtered TEC maps from the text file and loads it into a dataframe using the pandas library. It focuses on the North-American magnetic latitudes and longitude ranges and generates a 2D numpy array for each TEC map corresponding to the date & time value present in the text file. The TEC maps are processed by filling the missing TEC values due to missing magnetic latitude or longitude values to get a fixed shape of (75, 73) for all the TEC maps.  
* `filter_tec_map.py`: This file reads the TEC maps stored in the numpy array and fills the 'Nan' or missing TEC values with -1.  
* `create_tec_map_table.py`: This file creates a new sqlite3 database with filepath and datetime as the fields. For the given [start datetime, end datetime] range of the TEC maps, it iterates through all the datatime TEC maps from the start datetime and keeps track of the corresponding filepath of filled 2D numpy TEC array. This way we also track the missing TEC maps at a particular datetime.  
* `fill_missing_tec_map.py`: This file reads the sqlite3 database that was created and gets the datetime of the missing TEC maps and fills it using the nearest datetime TEC map available in the neighbouring window.

## Usage
Enter the appropriate start date and end date in all the files whereever required and run the commands in the order given below. 

1. Reading and generating the TEC maps in 2D numpy array.

    ```$ python generate_tec_map_files.py```
    
2. Filling the missing TEC values in the generated numpy arrays.

    $ python filter_tec_map.py  

3. Creating the sqlite3 database for tracking the created TEC arrays and missing TEC arrays.

    $ python create_tec_map_table.py

4. Using the database to fill the missing TEC arrays. 

    $ python fill_missing_tec_map.py
  
