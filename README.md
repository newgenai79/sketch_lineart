# Sketch/Lineart extractor

## Huggingface
 - https://huggingface.co/spaces/KBlueLeaf/Sketch-Gen

## Installation video

 - Youtube

## Installation steps

### Step 1: Clone the repository
```	
git clone https://github.com/newgenai79/sketch_lineart
```

### Step 2: Navigate inside the cloned repository
```	
cd sketch_lineart
```

### Step 3: Create virtual environment (tested on python 3.10.11)
```	
python -m venv venv
```

### Step 4: Activate virtual environment
```	
venv\scripts\activate
```

### Step 5: Install requirements
```	
pip install wheel
```
```
pip install -r requirements_windows.txt
```

### Step 6: Download models
```	
git clone https://huggingface.co/lllyasviel/paints_undo_single_frame lllyasviel/paints_undo_single_frame
```
	
### Step 7: Launch gradio based WebUI
```
venv\scripts\activate
```
```	
python app.py
```