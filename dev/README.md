<!-- markdownlint-disable MD033 -->

# How to make dummy data and train?
 
## Easiest way:
```bash
make run
```
It will run 
  make pull-> 
  make log_dir->
  make clear_data-> 
  make make_data-> 
  make train 
subsequently.

## Otherwise:
If you don't have data yet
```bash
make make_data
make log_dir
make train
```

If you have one
```bash
make clear_data
make make_data
make log_dir
make train
```

If you want to update the branch
```bash
make pull
```
