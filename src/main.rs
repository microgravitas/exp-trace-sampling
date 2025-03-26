#![allow(non_snake_case)]
#![allow(dead_code)]

mod sreach;

use itertools::Itertools;
use sreach::SReachTraceOracle;
use clap::{Parser, ValueEnum};
use fxhash::{FxHashMap, FxHashSet};
use graphbench::algorithms::LinearGraphAlgorithms;
use graphbench::degengraph::DegenGraph;
use graphbench::graph::*;
use graphbench::editgraph::EditGraph;
use graphbench::io::*;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[clap(author, version="1.0", about, long_about = None)]
struct Args {
    #[clap(value_enum)]
    query: QueryType,    

    #[clap(value_enum)]
    method: QueryMethod,

    #[clap(value_enum)]
    size: QuerySize,    

    number:usize,

    /// The network file
    file: String,

    // Random seed
    #[clap(short,long)]
    seed:Option<u64>,
}

#[derive(Clone, Debug, ValueEnum)]
enum QueryType {
    Trace,
    #[clap(name="trace_count")]
    TraceCount,
    Neighbours
}

#[derive(Clone, Debug, ValueEnum)]
enum QueryMethod {
    Basic,
    #[clap(name="sreach2")]
    SReach2
}

#[derive(Clone, Debug, ValueEnum)]
enum QuerySize {
    #[clap(name="const_10")]
    Const10,
    #[clap(name="const_50")]
    Const50,
    Log,
    Sqrt
}

const COUNT_MOD:usize = 100000;

fn main() -> Result<(), &'static str> {
    let args = Args::parse();
    let filename = args.file;

    // Load graph
    let mut graph = match EditGraph::from_file(&filename) {
        Ok(G) => G,
        Err(msg) => {
            println!("{msg}");
            return Err("Parsing error");
        }
    };

    graph.remove_loops();
    let graph = DegenGraph::from_graph(&graph);

    let degen = graph.left_degrees().into_values().max().expect("Graph is empty");
    let sreach2 = graph.sreach_sizes(2).into_values().max().unwrap();
    
    let n = graph.num_vertices();
    println!(
        "Loaded graph with n={}, m={}, d={}, s2={}",
        graph.num_vertices(),
        graph.num_edges(),
        degen,
        sreach2
    );

    let query_size:usize = match args.size {
        QuerySize::Const10 => 10,
        QuerySize::Const50 => 50,
        QuerySize::Log => (n.ilog2()+1) as usize,
        QuerySize::Sqrt => (n.isqrt()+1) as usize,
    };
    let num_queries = args.number;

    let large_deg_set = (query_size*10) as usize;

    // Only take large degree vertices
    let elements: Vec<u32> = graph.degrees().into_iter()
            .sorted_by_key(|(_,deg)| u32::MAX - deg)
            .map(|(u,_)| u).take(large_deg_set).collect_vec();

    // Take all vertices
    let elements: Vec<u32> = graph.vertices().cloned().collect();
    println!("Running {num_queries} queries of size {query_size} ({:?}) in large-degree set of size {}", args.size, elements.len());

    let rng = if let Some(seed) = args.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_os_rng()
    };

    let query_iter = RandomSetIter::new(elements, rng, query_size).take(num_queries);
    
    let total = match (args.method, args.query) {
        (QueryMethod::Basic, QueryType::Trace) => trace_basic(&graph, query_iter),
        (QueryMethod::Basic, QueryType::TraceCount) => trace_count_basic(&graph, query_iter),
        (QueryMethod::Basic, QueryType::Neighbours) => neighbours_basic(&graph, query_iter),
        (QueryMethod::SReach2, QueryType::Trace) => trace_sreach(&graph, query_iter),
        (QueryMethod::SReach2, QueryType::TraceCount) => trace_count_sreach(&graph, query_iter),
        (QueryMethod::SReach2, QueryType::Neighbours) => neighbours_sreach(&graph, query_iter),
    };

    println!("Debug counter value is {total}");

    Ok(())
}

pub fn trace_count_basic(graph:&DegenGraph, queries:impl Iterator<Item=Vec<u32>>) -> usize {
    let mut res = 0;
    for query in queries {
        let query_neighs:VertexSet = graph.neighbourhood(query.iter()).iter().cloned().collect();
        let query:VertexSet = query.into_iter().collect();
        let mut traces:FxHashMap<Vec<u32>,usize> = FxHashMap::default();
        for u in query_neighs {
            let u_neighs:VertexSet = graph.neighbours(&u).cloned().collect();
            let mut trace:Vec<Vertex> = query.intersection(&u_neighs).cloned().collect();
            trace.sort_unstable();
            *traces.entry(trace).or_default() += 1;
        }
        res = (res + traces.into_iter().map(|(trace,count)| (trace.len() * count) % COUNT_MOD ).sum::<usize>()) % COUNT_MOD;
    }
    res
}

pub fn trace_basic(graph:&DegenGraph, queries:impl Iterator<Item=Vec<u32>>) -> usize {
    let mut res = 0;
    for query in queries {
        let query_neighs:VertexSet = graph.neighbourhood(query.iter()).iter().cloned().collect();
        let query:VertexSet = query.into_iter().collect();
        let mut traces:FxHashSet<Vec<u32>> = FxHashSet::default();
        for u in query_neighs {
            let u_neighs:VertexSet = graph.neighbours(&u).cloned().collect();
            let mut trace:Vec<Vertex> = query.intersection(&u_neighs).cloned().collect();
            trace.sort_unstable();
            traces.insert(trace);
        }
        res = (res + traces.into_iter().map(|trace| trace.len() % COUNT_MOD ).sum::<usize>()) % COUNT_MOD;
    }
    res
}

pub fn neighbours_basic(graph:&DegenGraph, queries:impl Iterator<Item=Vec<u32>>) -> usize {
    let mut res = 0;
    for query in queries {
        res = (res + graph.neighbourhood(query.iter()).len()) % COUNT_MOD;
    }
    res
}

pub fn trace_sreach(graph:&DegenGraph, queries:impl Iterator<Item=Vec<u32>>) -> usize  {
    let mut res = 0;
    let oracle = SReachTraceOracle::for_graph(graph);
    println!("Oracle constructed");
    for mut query in queries {
        query.sort_unstable_by_key(|x| graph.index_of(x));
        let traces = oracle.compute_traces(&query, graph);
        res = (res + traces.into_iter().filter(|(_,count)| count > &0).map(|(trace,_)| trace.len() % COUNT_MOD ).sum::<usize>()) % COUNT_MOD;
    }
    res
}

pub fn trace_count_sreach(graph:&DegenGraph, queries:impl Iterator<Item=Vec<u32>>) -> usize  {
    let mut res = 0;
    let oracle = SReachTraceOracle::for_graph(graph);
    println!("Oracle constructed");
    for mut query in queries {
        query.sort_unstable_by_key(|x| graph.index_of(x));
        let traces = oracle.compute_traces(&query, graph);
        res = (res + traces.into_iter().map(|(trace,count)| (trace.len() * count) % COUNT_MOD ).sum::<usize>()) % COUNT_MOD;
    }
    res
}

pub fn neighbours_sreach(graph:&DegenGraph, queries:impl Iterator<Item=Vec<u32>>) -> usize  {
    let mut res = 0;
    let oracle = SReachTraceOracle::for_graph(graph);
    for mut query in queries {
        query.sort_unstable_by_key(|x| graph.index_of(x));
        res = (res + oracle.compute_neighbour_count(&query, graph)) % COUNT_MOD;
    }
    res
}


struct RandomSetIter {
    elements:Vec<u32>,
    current:Vec<u32>,
    current_indices:FxHashSet<usize>,
    size:usize,
    rng:StdRng
}

impl Iterator for RandomSetIter {
    type Item = Vec<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.current_indices.clear();
        self.current.clear();

        while self.current.len() < self.size {
            let i = self.rng.random_range(0..self.elements.len());
            if !self.current_indices.contains(&i) {
                self.current_indices.insert(i);
                self.current.push(self.elements[i]);
            }
        }

       Some(self.current.clone())
    }
}

impl RandomSetIter {
    pub fn new(elements:Vec<u32>, rng:StdRng, size:usize) -> Self {
        assert!(elements.len() >= size);
        let current = Vec::with_capacity(size);
        let current_indices = FxHashSet::default();
        RandomSetIter{ elements, current, current_indices, size, rng }
    }
}