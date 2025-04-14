# Word prediction, foundational ai pt 2


## setup your virtual environment. 
On Linux, i set it up like this:
```
python3 -m venv ./.venv
```
On windows, it is similar
```
python -m venv ./.venv
```
I assume mac will be similar to linox

## activate virtual environment
linux and macos
```
source .venv/bin/activate
```
windows
```
.venv/Scripts/activate
```


## install dependencies
```
python -m pip install -r requirements.txt
```

# usage

## create tokenizer

```
python ./tokenize_bpe.py
```

```
python ./word_prediction.py <model> <mode> [--state STATE] [--device DEVICE]
```


## training
```
python ./word_prediction.py (MODEL) train [--device DEVICE]
``` 
- Replace (MODEL) with either `rnn`,`lstm`, or `transformer`
- device is a torch device. can be cpu, cuda, mps


## testing
```
python ./word_prediction.py (MODEL) test --state (MODELPATH) --device (DEVICE)
``` 

- Replace (MODEL) with either `rnn`,`lstm`, or `transformer`
- (MODELPATH) is the trained model
- (DEVICE) is torch device. can be cpu, cuda, mps


## drawing
```
python ./word_prediction.py (MODEL) draw 
```
- Replace (MODEL) with either `rnn`,`lstm`, or `transformer`
- will produce a graph object and png, which is used in the report.
