use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ops::{Add, Deref, Neg, Sub};
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Instant};
use usiagent::command::{UsiInfoSubCommand, UsiScore, UsiScoreMate};
use usiagent::error::EventHandlerError;
use usiagent::event::{EventDispatcher, MapEventKind, UserEvent, UserEventDispatcher, UserEventKind, UserEventQueue, USIEventDispatcher};
use usiagent::hash::KyokumenHash;
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{LegalMove, Rule, State};
use usiagent::shogi::{MochigomaCollections, MochigomaKind, ObtainKind, Teban};
use crate::error::ApplicationError;
use crate::evalutor::Evalutor;
use crate::transposition_table::{TT, TTPartialEntry, ZobristHash};

pub const TURN_LIMIT:u32 = 10000;
pub const BASE_DEPTH:u32 = 10;
pub const MAX_DEPTH:u32 = 10;
pub const MAX_THREADS:u32 = 2;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Score {
    NEGINFINITE,
    Value(i32),
    INFINITE,
}
impl Neg for Score {
    type Output = Score;

    fn neg(self) -> Score {
        match self {
            Score::Value(v) => Score::Value(-v),
            Score::INFINITE => Score::NEGINFINITE,
            Score::NEGINFINITE => Score::INFINITE,
        }
    }
}
impl Add<i32> for Score {
    type Output = Self;

    fn add(self, other:i32) -> Self::Output {
        match self {
            Score::Value(v) => Score::Value(v + other),
            Score::INFINITE => Score::INFINITE,
            Score::NEGINFINITE => Score::NEGINFINITE,
        }
    }
}
impl Sub<i32> for Score {
    type Output = Self;

    fn sub(self, other:i32) -> Self::Output {
        match self {
            Score::Value(v) => Score::Value(v - other),
            Score::INFINITE => Score::INFINITE,
            Score::NEGINFINITE => Score::NEGINFINITE,
        }
    }
}
impl Default for Score {
    fn default() -> Self {
        Score::NEGINFINITE
    }
}
pub struct Environment<L,S> where L: Logger, S: InfoSender {
    pub event_queue:Arc<Mutex<UserEventQueue>>,
    pub info_sender:S,
    pub on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
    pub hasher:Arc<KyokumenHash<u64>>,
    pub turn_limit:Option<Instant>,
    pub base_depth:u32,
    pub max_depth:u32,
    pub max_threads:u32,
    pub abort:Arc<AtomicBool>,
    pub stop:Arc<AtomicBool>,
    pub quited:Arc<AtomicBool>,
    pub transposition_table:Arc<TT<u64,Score,{1<<20},4>>,
    pub nodes:Arc<AtomicU64>
}
impl<L,S> Clone for Environment<L,S> where L: Logger, S: InfoSender {
    fn clone(&self) -> Self {
        Environment {
            event_queue:Arc::clone(&self.event_queue),
            info_sender:self.info_sender.clone(),
            on_error_handler:Arc::clone(&self.on_error_handler),
            hasher:Arc::clone(&self.hasher),
            turn_limit:self.turn_limit.clone(),
            base_depth:self.base_depth,
            max_depth:self.max_depth,
            max_threads:self.max_threads,
            abort:Arc::clone(&self.abort),
            stop:Arc::clone(&self.stop),
            quited:Arc::clone(&self.quited),
            transposition_table:self.transposition_table.clone(),
            nodes:Arc::clone(&self.nodes),
        }
    }
}
#[derive(Debug)]
pub enum EvaluationResult {
    Immediate(Score, VecDeque<LegalMove>, ZobristHash<u64>),
    Timeout(Option<(Score, VecDeque<LegalMove>, ZobristHash<u64>)>)
}
impl<L,S> Environment<L,S> where L: Logger, S: InfoSender {
    pub fn new(event_queue:Arc<Mutex<UserEventQueue>>,
               info_sender:S,
               on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
               hasher:Arc<KyokumenHash<u64>>,
               turn_limit:Option<Instant>,
               base_depth:u32,
               max_depth:u32,
               max_threads:u32,
               transposition_table: &Arc<TT<u64,Score,{1 << 20},4>>
    ) -> Environment<L,S> {
        let abort = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let quited = Arc::new(AtomicBool::new(false));

        Environment {
            event_queue:event_queue,
            info_sender:info_sender,
            on_error_handler:on_error_handler,
            hasher:hasher,
            turn_limit:turn_limit,
            base_depth:base_depth,
            max_depth:max_depth,
            max_threads:max_threads,
            abort:abort,
            stop:stop,
            quited:quited,
            transposition_table:Arc::clone(transposition_table),
            nodes:Arc::new(AtomicU64::new(0))
        }
    }
}
pub struct GameState<'a> {
    pub teban:Teban,
    pub state:&'a Arc<State>,
    pub alpha:Score,
    pub beta:Score,
    pub m:Option<LegalMove>,
    pub mc:&'a Arc<MochigomaCollections>,
    pub zh:ZobristHash<u64>,
    pub depth:u32,
    pub current_depth:u32,
    pub base_depth:u32,
    pub max_depth:u32
}
pub struct Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    l:PhantomData<L>,
    s:PhantomData<S>,
    receiver:Receiver<Result<EvaluationResult, ApplicationError>>,
    sender:Sender<Result<EvaluationResult, ApplicationError>>
}
pub trait Search<L,S>: Sized where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                     evalutor: &Arc<Evalutor>) -> Result<EvaluationResult,ApplicationError>;
    fn timelimit_reached(&self,env:&mut Environment<L,S>) -> bool {
        env.turn_limit.map(|l| Instant::now() >= l).unwrap_or(false)
    }

    fn send_info(&self, env:&mut Environment<L,S>,
                 depth:u32, seldepth:u32, pv:&VecDeque<LegalMove>, score:&Score) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

        let mut commands: Vec<UsiInfoSubCommand> = Vec::new();

        match score {
            Score::INFINITE => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Plus)))
            },
            Score::NEGINFINITE => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Minus)))
            },
            Score::Value(s) => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Cp(*s as i64)))
            }
        }

        commands.push(UsiInfoSubCommand::Depth(depth));

        if depth < seldepth {
            commands.push(UsiInfoSubCommand::SelDepth(seldepth));
        }

        if pv.len() > 0 {
            commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
            commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));
        }

        env.info_sender.send(commands)?;
        Ok(env.info_sender.flush()?)
    }

    fn update_tt<'a>(&self, env: &mut Environment<L, S>,
                     zh: &'a ZobristHash<u64>,
                     depth: u32,
                     score: Score) {
        let mut tte = env.transposition_table.entry(&zh);
        let tte = tte.or_default();

        if tte.depth < depth as i8 - 1 {
            tte.depth = depth as i8 - 1;
            tte.score = score;
        }
    }

    fn update_best_move<'a>(&self, env: &mut Environment<L, S>,
                            zh: &'a ZobristHash<u64>,
                            depth: u32,
                            score:Score,
                            m: Option<LegalMove>) {
        let mut tte = env.transposition_table.entry(zh);
        let tte = tte.or_default();

        if tte.depth < depth as i8 || (tte.depth == depth as i8 && tte.score < score) {
            tte.depth = depth as i8;
            tte.score = score;
            tte.best_move = m;
        }
    }
}
impl<L,S> Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    pub fn new() -> Root<L,S> {
        let(s,r) = mpsc::channel();

        Root {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
            receiver:r,
            sender:s
        }
    }

    pub fn create_event_dispatcher<'a,T>(on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,stop:&Arc<AtomicBool>,quited:&Arc<AtomicBool>)
                                         -> UserEventDispatcher<'a,T,ApplicationError,L> {

        let mut event_dispatcher = USIEventDispatcher::new(&on_error_handler);

        {
            let stop = stop.clone();

            event_dispatcher.add_handler(UserEventKind::Stop, move |_,e| {
                match e {
                    &UserEvent::Stop => {
                        stop.store(true,atomic::Ordering::Release);
                        Ok(())
                    },
                    e => Err(EventHandlerError::InvalidState(e.event_kind())),
                }
            });
        }

        {
            let stop = stop.clone();
            let quited = quited.clone();

            event_dispatcher.add_handler(UserEventKind::Quit, move |_,e| {
                match e {
                    &UserEvent::Quit => {
                        quited.store(true,atomic::Ordering::Release);
                        stop.store(true,atomic::Ordering::Release);
                        Ok(())
                    },
                    e => Err(EventHandlerError::InvalidState(e.event_kind())),
                }
            });
        }

        event_dispatcher
    }

    pub fn termination<'a,'b>(&self,
                              env:&mut Environment<L,S>,
                              gs: &mut GameState<'a>,
                              score:Score,
                              is_timeout:bool,
                              max_threads:u32,
                              mut threads:u32,
                              mut best_moves:VecDeque<LegalMove>) -> Result<EvaluationResult,ApplicationError> {
        env.abort.store(true,Ordering::Release);

        let mut score = score;
        let mut last_error = None;

        while threads < max_threads {
            match self.receiver.recv().map_err(|e| ApplicationError::from(e)).and_then(|r| r) {
                Ok(EvaluationResult::Immediate(s,mvs,zh)) |
                Ok(EvaluationResult::Timeout(Some((s,mvs,zh)))) => {
                    self.update_tt(env,&gs.zh,gs.depth,s);

                    if -s > score {
                        score = -s;
                        best_moves = mvs;

                        self.update_best_move(env,&zh,gs.depth - 1, score, best_moves.front().cloned());

                        if let Err(e) =  self.send_info(env, gs.base_depth,gs.current_depth,&best_moves,&score) {
                            last_error = Some(Err(e));
                        }
                    }
                    threads += 1;
                },
                e @ Err(_) => {
                    threads += 1;
                    last_error = Some(e);
                }
                _ => {
                    threads += 1;
                }
            }
        }

        if let Some(e) = last_error {
            e
        } else if is_timeout && best_moves.is_empty() {
            Ok(EvaluationResult::Timeout(None))
        } else {
            Ok(EvaluationResult::Immediate(score, best_moves,gs.zh.clone()))
        }
    }

    fn parallelized<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                           event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                           evalutor: &Arc<Evalutor>,
                           best_moves:VecDeque<LegalMove>) -> Result<EvaluationResult,ApplicationError>  {
        let mut gs = gs;
        let mut best_moves = best_moves;

        let mut mvs = Rule::legal_moves_all(gs.teban,&gs.state,&gs.mc);

        if let Some(TTPartialEntry {
                        depth: _,
                        score: _,
                        best_move: m
                    }) = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone()) {
            m.map(|m| mvs.insert(0,m));
        }

        let mut alpha = gs.alpha;
        let beta = gs.beta;
        let mut scoreval = Score::NEGINFINITE;

        let mut is_timeout = false;

        let mvs_count = mvs.len() as u64;

        let mut threads = env.max_threads.min(mvs_count as u32);
        let mut force_recv = false;

        let sender = self.sender.clone();

        let mut it = mvs.into_iter();

        loop {
            if threads == 0 || force_recv {
                let r = self.receiver.recv();

                let r = r?.map_err(|e| ApplicationError::from(e))?;

                threads += 1;

                match r {
                    EvaluationResult::Immediate(s, mvs,zh) => {
                        self.update_tt(env,&zh,gs.depth,s);

                        let s = -s;

                        if s > scoreval {
                            scoreval = s;

                            best_moves = mvs;

                            self.send_info(env, gs.base_depth, gs.current_depth, &best_moves, &scoreval)?;

                            self.update_best_move(env,&gs.zh,gs.depth,scoreval,best_moves.front().cloned());

                            if scoreval >= beta {
                                break;
                            }

                            if alpha < scoreval {
                                alpha = scoreval;
                            }
                        }

                        if env.stop.load(atomic::Ordering::Acquire) {
                            if best_moves.is_empty() {
                                is_timeout = true;
                            }
                            break;
                        }
                    },
                    EvaluationResult::Timeout(Some((s,mvs,_))) => {
                        if -s > scoreval {
                            scoreval = -s;
                            best_moves = mvs;
                            self.send_info(env, gs.base_depth, gs.current_depth, &best_moves, &scoreval)?;
                        }
                        is_timeout = true;
                        break;
                    },
                    EvaluationResult::Timeout(None) => {
                        is_timeout = true;
                        break;
                    }
                }

                let event_queue = Arc::clone(&env.event_queue);
                event_dispatcher.dispatch_events(self, &*event_queue)?;

                if env.stop.load(atomic::Ordering::Acquire) {
                    is_timeout = true;
                    break;
                }
            } else if let Some(m) = it.next() {
                let o = match m {
                    LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                    _ => None
                };

                let mut depth = gs.depth;

                if o.is_some() {
                    depth += 1;
                }

                let zh = gs.zh.updated(&env.hasher,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);

                let next = Rule::apply_move_none_check(&gs.state, gs.teban, gs.mc, m.to_applied_move());

                match next {
                    (state, mc, _) => {
                        let teban = gs.teban;
                        let state = Arc::new(state);
                        let mc = Arc::new(mc);
                        let alpha = alpha;
                        let beta = beta;
                        let current_depth = gs.current_depth;
                        let base_depth = gs.base_depth;
                        let max_depth = gs.max_depth;

                        let mut env = env.clone();
                        let evalutor = Arc::clone(evalutor);

                        let sender = sender.clone();

                        let b = std::thread::Builder::new();

                        let sender = sender.clone();

                        let _ = b.stack_size(1024 * 1024 * 200).spawn(move || {
                            let mut event_dispatcher = Self::create_event_dispatcher::<Recursive<L, S>>(&env.on_error_handler, &env.stop, &env.quited);

                            let mut gs = GameState {
                                teban: teban.opposite(),
                                state: &state,
                                alpha: -beta,
                                beta: -alpha,
                                m: Some(m),
                                mc: &mc,
                                zh: zh.clone(),
                                depth: depth - 1,
                                current_depth: current_depth + 1,
                                base_depth:base_depth,
                                max_depth:max_depth
                            };

                            let strategy = Recursive::new();

                            let r = strategy.search(&mut env, &mut gs, &mut event_dispatcher, &evalutor);

                            let _ = sender.send(r);
                        });

                        threads -= 1;
                    }
                }
            } else if threads == env.max_threads.min(mvs_count as u32) {
                force_recv = true;
            } else {
                break;
            }
        }

        if scoreval == Score::NEGINFINITE {
            self.send_info(env, gs.base_depth, gs.current_depth, &best_moves, &scoreval)?;
        }

        self.termination(env, &mut gs,scoreval,is_timeout, env.max_threads.min(mvs_count as u32), threads, best_moves)
    }
}
impl<L,S> Search<L,S> for Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                     evalutor: &Arc<Evalutor>) -> Result<EvaluationResult,ApplicationError> {
        let base_depth = gs.depth.min(env.base_depth);
        let mut depth = 1;
        let mut best_moves = VecDeque::new();
        let mut result = None;

        loop {
            env.abort.store(false,Ordering::Release);

            gs.depth = depth;
            gs.base_depth = depth;
            gs.max_depth = env.max_depth - (base_depth - depth);

            let current_result = self.parallelized(env, gs, event_dispatcher, evalutor, best_moves.clone())?;

            depth += 1;

            match current_result {
                EvaluationResult::Immediate(s,mvs,zh) if base_depth + 1 == depth && mvs.len() > 0 => {
                    return Ok(EvaluationResult::Immediate(s,mvs,zh));
                },
                EvaluationResult::Immediate(_,_,_) if base_depth + 1 == depth => {
                    return Ok(result.unwrap_or(EvaluationResult::Timeout(None)));
                },
                EvaluationResult::Immediate(s,mvs,zh) if mvs.len() > 0 => {
                    best_moves = mvs.clone();
                    result = Some(EvaluationResult::Immediate(s,mvs,zh));
                },
                EvaluationResult::Immediate(_,_,_) => (),
                EvaluationResult::Timeout(Some((_,mvs,_))) if mvs.is_empty() => {
                    return Ok(result.unwrap_or(EvaluationResult::Timeout(None)));
                },
                EvaluationResult::Timeout(Some((s,mvs,zh))) => {
                    return Ok(EvaluationResult::Immediate(s,mvs,zh));
                },
                EvaluationResult::Timeout(None) => {
                    return Ok(result.unwrap_or(EvaluationResult::Timeout(None)));
                }
            }
        }
    }
}
pub struct Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    l:PhantomData<L>,
    s:PhantomData<S>,
}
impl<L,S> Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    pub fn new() -> Recursive<L,S> {
        Recursive {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
        }
    }
}
impl<L,S> Search<L,S> for Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a, 'b>(&self, env: &mut Environment<L, S>, gs: &mut GameState<'a>,
                      event_dispatcher: &mut UserEventDispatcher<'b, Recursive<L, S>, ApplicationError, L>,
                      evalutor: &Arc<Evalutor>) -> Result<EvaluationResult, ApplicationError> {
        env.nodes.fetch_add(1,Ordering::Release);

        if self.timelimit_reached(env) || env.abort.load(Ordering::Acquire) || env.stop.load(Ordering::Acquire) {
            return Ok(EvaluationResult::Timeout(None));
        }

        let prev_move = gs.m.ok_or(ApplicationError::LogicError(String::from(
            "move is not set."
        )))?;

        let obtained = match prev_move {
            LegalMove::To(m) => m.obtained(),
            _ => None
        };

        if let Some(ObtainKind::Ou) = obtained {
            let mut mvs = VecDeque::new();

            mvs.push_front(prev_move);

            return Ok(EvaluationResult::Immediate(Score::NEGINFINITE,mvs,gs.zh.clone()));
        }

        if Rule::is_mate(gs.teban,&gs.state) {
            let mut mvs = VecDeque::new();

            mvs.push_front(prev_move);

            return Ok(EvaluationResult::Immediate(Score::INFINITE,mvs,gs.zh.clone()));
        }

        if gs.depth == 0 || gs.current_depth >= gs.max_depth {
            let s = Score::Value(evalutor.evalute(gs.teban,&gs.state.get_banmen(),&gs.mc));

            let mut mvs = VecDeque::new();

            mvs.push_front(prev_move);

            return Ok(EvaluationResult::Immediate(s,mvs,gs.zh.clone()))
        }

        let mut mvs = Rule::legal_moves_all(gs.teban,&gs.state,&gs.mc);

        if let Some(TTPartialEntry {
                        depth: _,
                        score: _,
                        best_move: m
                    }) = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone()) {
            m.map(|m| mvs.insert(0,m));
        }

        let mut alpha = gs.alpha;
        let beta = gs.beta;
        let mut scoreval = Score::NEGINFINITE;
        let mut best_moves = VecDeque::new();

        for m in mvs {
            let o = match m {
                LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                _ => None
            };

            let mut depth = gs.depth;

            if o.is_some() {
                depth += 1;
            }

            let zh = gs.zh.updated(&env.hasher,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);

            let next = Rule::apply_move_none_check(&gs.state, gs.teban, gs.mc, m.to_applied_move());

            match next {
                (state, mc, _) => {
                    let state = Arc::new(state);
                    let mc = Arc::new(mc);
                    let prev_zh = gs.zh.clone();
                    let d = gs.depth;

                    let mut gs = GameState {
                        teban: gs.teban.opposite(),
                        state: &state,
                        alpha: -gs.beta,
                        beta: -alpha,
                        m: Some(m),
                        mc: &mc,
                        zh: zh.clone(),
                        depth: depth - 1,
                        current_depth: gs.current_depth + 1,
                        base_depth: gs.base_depth,
                        max_depth:gs.max_depth
                    };

                    let strategy = Recursive::new();

                    match strategy.search(env, &mut gs, event_dispatcher, evalutor)? {
                        EvaluationResult::Immediate(s, mvs, zh) => {
                            self.update_tt(env,&zh,gs.depth,s);

                            let s = -s;

                            if s > scoreval || s == Score::INFINITE {
                                scoreval = s;

                                best_moves = mvs;

                                self.update_best_move(env,&prev_zh,d,scoreval,Some(m));

                                if scoreval >= beta {
                                    best_moves.push_front(prev_move);
                                    return Ok(EvaluationResult::Immediate(scoreval, best_moves, prev_zh.clone()));
                                }
                            }

                            if alpha < s {
                                alpha = s;
                            }
                        },
                        EvaluationResult::Timeout(Some((s,mvs,zh))) => {
                            self.update_tt(env,&zh,gs.depth,s);

                            let s = -s;

                            if s > scoreval {
                                scoreval = s;

                                best_moves = mvs;

                                self.update_best_move(env,&prev_zh,d,scoreval,Some(m));
                            }

                            best_moves.push_front(prev_move);
                            return Ok(EvaluationResult::Timeout(Some((s,best_moves,prev_zh))));
                        },
                        EvaluationResult::Timeout(None) if best_moves.len() > 0 => {
                            best_moves.push_front(prev_move);
                            return Ok(EvaluationResult::Timeout(Some((scoreval,best_moves,prev_zh))));
                        },
                        EvaluationResult::Timeout(None) => {
                            return Ok(EvaluationResult::Timeout(None));
                        },
                    }

                    event_dispatcher.dispatch_events(self, &*env.event_queue)?;

                    if env.abort.load(Ordering::Acquire) || env.stop.load(atomic::Ordering::Acquire) {
                        if best_moves.is_empty() {
                            return Ok(EvaluationResult::Timeout(None));
                        } else {
                            best_moves.push_front(prev_move);
                            return Ok(EvaluationResult::Timeout(Some((scoreval, best_moves, prev_zh.clone()))));
                        }
                    }
                }
            }
        }

        best_moves.push_front(prev_move);
        Ok(EvaluationResult::Immediate(scoreval, best_moves,gs.zh.clone()))
    }
}