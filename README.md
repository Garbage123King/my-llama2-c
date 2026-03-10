pre:
```
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

# MY RUN
```
gcc main.c -o main -lm
./main
```

# ORIGINAL RUN
run:
```
./make.sh
./run stories42M.bin
```
prompt:
```
./run stories42M.bin -i "A boy was hit by a cat"
```
