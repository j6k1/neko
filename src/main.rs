extern crate rand;
extern crate rand_distr;
extern crate rand_xorshift;
extern crate statrs;
extern crate parking_lot;

extern crate usiagent;
extern crate core;

use usiagent::output::USIStdErrorWriter;
use usiagent::UsiAgent;
use crate::error::ApplicationError;
use crate::player::Neko;

pub mod evalutor;
pub mod transposition_table;
pub mod error;
pub mod search;
pub mod player;

fn main() {
    match run() {
        Ok(()) => (),
        Err(ref e) =>  {
            let _ = USIStdErrorWriter::write(&e.to_string());
        }
    };
}
fn run() -> Result<(),ApplicationError> {
    let agent = UsiAgent::new(Neko::new());

    let r = agent.start_default(|on_error_handler, e| {
        match on_error_handler {
            Some(ref h) => {
                let _ = h.lock().map(|h| h.call(e));
            },
            None => (),
        }
    });
    r.map_err(|_| ApplicationError::AgentRunningError(String::from(
        "An error occurred while running USIAgent. See log for details..."
    )))
}