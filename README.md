Experimental program to compare different approaches to compute the trace, trace frequency, and neighbourhood size of a query set in a network.

## To run the program

1. Install the latest version of [rust](https://www.rust-lang.org/tools/install)
2. Prepare a network in a suitable file format (see below) or use one from the [network corpus](https://github.com/microgravitas/network-corpus)
3. Run the following command to run the program on a specific network
```
cargo run --release <QUERY> <METHOD> <SIZE> <NUMBER> <FILE>
```
For example, to run the trace counting on the `karate.txt.gz` network for 100 random query sets of size 10 using our method:
```
cargo run --release trace_count sreach2 const_10 100 karate.txt.gz

```
To see all possible values for the different parameters, run
```
cargo run --release -- --help
```
Note the double dash separator before `--help`.

## File format

The program accepts text files to gzipped text files. The file should only contain a list of edges encoded as integers, separated by line breaks. For example, a simple triangle graph is given by
```
0 1
1 2
2 0
```

