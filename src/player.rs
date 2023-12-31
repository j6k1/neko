use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use usiagent::command::{BestMove, CheckMate, UsiInfoSubCommand, UsiOptType};
use usiagent::error::{PlayerError, UsiProtocolError};
use usiagent::event::{GameEndState, SysEventOption, SysEventOptionKind, UserEvent, UserEventQueue, UsiGoMateTimeLimit, UsiGoTimeLimit};
use usiagent::hash::{KyokumenHash};
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::output::USIOutputWriter;
use usiagent::player::{InfoSender, OnKeepAlive, PeriodicallyInfo, USIPlayer};
use usiagent::rule::{AppliedMove, Kyokumen, State};
use usiagent::shogi::{Banmen, Mochigoma, MochigomaCollections, Move, Teban};
use crate::error::ApplicationError;
use crate::evalutor::Evalutor;
use crate::search::{BASE_DEPTH, Environment, EvaluationResult, GameState, MAX_DEPTH, MAX_THREADS, Root, Score, Search, TURN_LIMIT};
use crate::transposition_table::{TT, ZobristHash};

pub trait FromOption {
    fn from_option(option:SysEventOption) -> Option<Self> where Self: Sized;
}
impl FromOption for i64 {
    fn from_option(option: SysEventOption) -> Option<i64> {
        match option {
            SysEventOption::Num(v) => Some(v),
            _ => None
        }
    }
}
impl FromOption for u32 {
    fn from_option(option: SysEventOption) -> Option<u32> {
        match option {
            SysEventOption::Num(v) => Some(v as u32),
            _ => None
        }
    }
}
impl FromOption for usize {
    fn from_option(option: SysEventOption) -> Option<usize> {
        match option {
            SysEventOption::Num(v) => Some(v as usize),
            _ => None
        }
    }
}
impl FromOption for bool {
    fn from_option(option: SysEventOption) -> Option<bool> {
        match option {
            SysEventOption::Bool(b) => Some(b),
            _ => None
        }
    }
}
pub struct Neko {
    evalutor: Arc<Evalutor>,
    kyokumen:Option<Kyokumen>,
    zh:Option<ZobristHash<u64>>,
    hasher:Arc<KyokumenHash<u64>>,
    transposition_table:Arc<TT<u64,Score,{1<<20},4>>,
    base_depth:u32,
    max_depth:u32,
    max_threads:u32,
    turn_limit:Option<u32>,
}
impl fmt::Debug for Neko {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Neko")
    }
}
impl Neko {
    pub fn new() -> Neko {
        Neko {
            evalutor:Arc::new(Evalutor::new()),
            kyokumen:None,
            zh:None,
            hasher:Arc::new(KyokumenHash::new()),
            transposition_table:Arc::new(TT::new()),
            base_depth:BASE_DEPTH,
            max_depth:MAX_DEPTH,
            max_threads:MAX_THREADS,
            turn_limit:None,
        }
    }
}
impl USIPlayer<ApplicationError> for Neko {
    const ID: &'static str = "neko";
    const AUTHOR: &'static str = "j6k1";

    fn get_option_kinds(&mut self) -> Result<BTreeMap<String,SysEventOptionKind>,ApplicationError> {
        let mut kinds:BTreeMap<String,SysEventOptionKind> = BTreeMap::new();
        kinds.insert(String::from("USI_Hash"),SysEventOptionKind::Num);
        kinds.insert(String::from("USI_Ponder"),SysEventOptionKind::Bool);
        kinds.insert(String::from("MaxDepth"),SysEventOptionKind::Num);
        kinds.insert(String::from("Threads"),SysEventOptionKind::Num);
        kinds.insert(String::from("BaseDepth"),SysEventOptionKind::Num);
        kinds.insert(String::from("TurnLimit"),SysEventOptionKind::Num);

        Ok(kinds)
    }

    fn get_options(&mut self) -> Result<BTreeMap<String,UsiOptType>,ApplicationError> {
        let mut options:BTreeMap<String,UsiOptType> = BTreeMap::new();
        options.insert(String::from("BaseDepth"),UsiOptType::Spin(1,100,Some(BASE_DEPTH as i64)));
        options.insert(String::from("MaxDepth"),UsiOptType::Spin(1,100,Some(MAX_DEPTH as i64)));
        options.insert(String::from("Threads"),UsiOptType::Spin(1,1024,Some(MAX_THREADS as i64)));
        options.insert(String::from("TurnLimit"),UsiOptType::Spin(1,3600000,Some(TURN_LIMIT as i64)));

        Ok(options)
    }

    fn take_ready<W,L>(&mut self, _:OnKeepAlive<W,L>)
                       -> Result<(),ApplicationError> where W: USIOutputWriter + Send + 'static,
                                                            L: Logger + Send + 'static {
        Ok(())
    }

    fn set_option(&mut self,name:String,value:SysEventOption) -> Result<(),ApplicationError> {
        match &*name {
            "MaxDepth" => {
                self.max_depth = u32::from_option(value).unwrap_or(MAX_DEPTH);
            },
            "BaseDepth" => {
                self.base_depth = u32::from_option(value).unwrap_or(BASE_DEPTH);
            },
            "Threads" => {
                self.max_threads = u32::from_option(value).unwrap_or(MAX_THREADS);
            },
            "TurnLimit" => {
                self.turn_limit = u32::from_option(value);
            },
            _ => ()
        }

        Ok(())
    }

    fn newgame(&mut self) -> Result<(),ApplicationError> {
        self.kyokumen = None;

        match Arc::get_mut(&mut self.transposition_table) {
            Some(transposition_table) => {
                transposition_table.clear();
            },
            None => {
                return Err(ApplicationError::InvalidStateError(String::from(
                    "Failed to get mutable reference for transposition_table."
                )));
            }
        }
        Ok(())
    }
    fn set_position(&mut self,teban:Teban,banmen:Banmen,
                    ms:Mochigoma,mg:Mochigoma,_:u32,m:Vec<Move>)
                    -> Result<(),ApplicationError> {
        let zh = ZobristHash::new(&self.hasher,teban,&banmen,&ms,&mg);

        let teban = teban;
        let state = State::new(banmen);

        let mc = MochigomaCollections::new(ms,mg);

        let (t,state,mc,r) = self.apply_moves(state,teban, mc,&m.into_iter()
            .map(|m| m.to_applied_move())
            .collect::<Vec<AppliedMove>>(),
                                              zh,
                                              |_,t,banmen,mc,m,o,r| {
                                                  let mut zh = r;

                                                  let zh = match m {
                                                      &Some(m) => {
                                                          zh = zh.updated(&self.hasher,t,&banmen,&mc,m,&o);
                                                          zh
                                                      },
                                                      &None => {
                                                          zh
                                                      }
                                                  };
                                                  zh
                                              });

        let zh = r;

        self.kyokumen = Some(Kyokumen {
            state:state,
            mc:mc,
            teban:t
        });
        self.zh = Some(zh);
        Ok(())
    }

    fn think<L,S,P>(&mut self,think_start_time:Instant,
                    limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<UserEventQueue>>,
                    info_sender:S,periodically_info:P,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
                    -> Result<BestMove,ApplicationError>
        where L: Logger + Send + 'static,
              S: InfoSender,
              P: PeriodicallyInfo {
        let (teban,state,mc) = self.kyokumen.as_ref().map(|k| (k.teban,&k.state,&k.mc)).ok_or(
            UsiProtocolError::InvalidState(
                String::from("Position information is not initialized."))
        )?;

        let limit = limit.to_instant(teban,think_start_time);

        let mut env = Environment::new(
            Arc::clone(&event_queue),
            info_sender.clone(),
            Arc::clone(&on_error_handler),
            Arc::clone(&self.hasher),
            limit,
            self.turn_limit.map(|l| think_start_time + Duration::from_millis(l as u64)),
            self.base_depth,
            self.max_depth,
            self.max_threads,
            &self.transposition_table
        );

        let base_depth = env.base_depth;

        let mut event_dispatcher = Root::<L,S>::create_event_dispatcher(&on_error_handler,&env.stop,&env.quited);

        let _pinfo_sender = {
            let nodes = env.nodes.clone();
            let think_start_time = think_start_time.clone();
            let on_error_handler = env.on_error_handler.clone();

            periodically_info.start(1000,move || {
                let mut commands = vec![];
                commands.push(UsiInfoSubCommand::Nodes(nodes.load(Ordering::Acquire)));

                let sec = (Instant::now() - think_start_time).as_secs();

                if sec > 0 {
                    commands.push(UsiInfoSubCommand::Nps(nodes.load(Ordering::Acquire) / sec));
                }

                commands
            }, &on_error_handler)
        };

        let zh = match self.zh.as_ref() {
            Some(zh) => zh.clone(),
            None => {
                return Err(ApplicationError::InvalidStateError(format!("ZobristHash is not initialized!")))
            }
        };

        let mut gs = GameState {
            teban: teban,
            state: &Arc::new(state.clone()),
            alpha: Score::NEGINFINITE,
            beta: Score::INFINITE,
            m:None,
            mc: &Arc::new(mc.clone()),
            zh:zh,
            depth:base_depth,
            current_depth:0,
            base_depth:base_depth,
            max_depth:env.max_depth
        };

        let strategy  = Root::new();

        let evalutor = Arc::clone(&self.evalutor);

        let result = strategy.search(&mut env,&mut gs, &mut event_dispatcher, &evalutor);

        let bestmove = match result {
            Err(ref e) => {
                let _ = env.on_error_handler.lock().map(|h| h.call(e));
                BestMove::Resign
            },
            Ok(EvaluationResult::Timeout) => {
                BestMove::Resign
            },
            Ok(EvaluationResult::Immediate(Score::NEGINFINITE,_,_)) => {
                BestMove::Resign
            },
            Ok(EvaluationResult::Immediate(_,mvs,_)) if mvs.len() == 0 => {
                BestMove::Resign
            },
            Ok(EvaluationResult::Immediate(_,mvs,_)) => {
                BestMove::Move(mvs[0].to_move(),None)
            }
        };

        Ok(bestmove)
    }

    fn think_ponder<L,S,P>(&mut self,_:&UsiGoTimeLimit,_:Arc<Mutex<UserEventQueue>>,
                           _:S,_:P,_:Arc<Mutex<OnErrorHandler<L>>>)
                           -> Result<BestMove,ApplicationError> where L: Logger + Send + 'static, S: InfoSender,
                                                                      P: PeriodicallyInfo + Send + 'static {
        unimplemented!();
    }

    fn think_mate<L,S,P>(&mut self,_:&UsiGoMateTimeLimit,_:Arc<Mutex<UserEventQueue>>,
                         _:S,_:P,_:Arc<Mutex<OnErrorHandler<L>>>)
                         -> Result<CheckMate,ApplicationError>
        where L: Logger + Send + 'static,
              S: InfoSender,
              P: PeriodicallyInfo {
        Ok(CheckMate::NotiImplemented)
    }

    fn on_stop(&mut self,_:&UserEvent) -> Result<(), ApplicationError> where ApplicationError: PlayerError {
        Ok(())
    }

    fn gameover<L>(&mut self,_:&GameEndState,
                   _:Arc<Mutex<UserEventQueue>>, _:Arc<Mutex<OnErrorHandler<L>>>) -> Result<(),ApplicationError> where L: Logger, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        Ok(())
    }

    fn on_ponderhit(&mut self,_:&UserEvent) -> Result<(), ApplicationError> where ApplicationError: PlayerError {
        Ok(())
    }

    fn on_quit(&mut self,_:&UserEvent) -> Result<(), ApplicationError> where ApplicationError: PlayerError {
        Ok(())
    }

    fn quit(&mut self) -> Result<(),ApplicationError> {
        Ok(())
    }
}
