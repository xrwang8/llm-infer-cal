fn main() {
    let exit = llm_infer_cal::run_with_args(std::env::args_os());
    print!("{}", exit.stdout);
    eprint!("{}", exit.stderr);
    std::process::exit(exit.code);
}
