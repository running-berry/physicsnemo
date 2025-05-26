<!-- markdownlint-disable MD033 -->

# How to make dummy data and train?
 
## Easiest way:
```bash
make run
```
It will run 
  make pull-> 
  make clear_data-> 
  make make_data-> 
  make train 
subsequently.

## Otherwise:
If you don't have data yet
```bash
make data
make train
```

If you have one
```bash
make clear_data
make data
make train
```

If you want to update the branch
```bash
make pull
```
