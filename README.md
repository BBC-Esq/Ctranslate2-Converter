# Ctranslate2-Converter
Tool to easily convert models into the Ctranslate2 format:

#### 1 - create virtual environment
```
python -m venv .
```

#### 2 - activate virtual environment
```
.\Scripts\activate
```

#### 3 - run setup script
> Installation scripts for other platforms are coming soon.
```
python setup_windows.py
```

#### 4 - run program
```
python convert_ctranslate2.py
```

## Usage

Select the folder containing the model files, select one or more quantizations, select the output directory, click "run."

The program will create a new appropriately-named folder in the selected output directory and the ```Ctranslate2``` model file ready to use.
