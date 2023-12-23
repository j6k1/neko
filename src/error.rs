use std::{error, fmt};
use std::sync::mpsc::RecvError;
use usiagent::error::{EventDispatchError, InfoSendError, PlayerError, UsiProtocolError};
use usiagent::event::{EventQueue, UserEvent, UserEventKind};

#[derive(Debug)]
pub enum ApplicationError {
    AgentRunningError(String),
    LogicError(String),
    InvalidStateError(String),
    EventDispatchError(String),
    InfoSendError(InfoSendError),
    RecvError(RecvError),
    UsiProtocolError(UsiProtocolError),
}
impl fmt::Display for ApplicationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ApplicationError::AgentRunningError(ref s) => write!(f, "{}",s),
            ApplicationError::LogicError(ref s) => write!(f,"{}",s),
            ApplicationError::InvalidStateError(ref s) => write!(f,"{}",s),
            ApplicationError::EventDispatchError(ref s) => write!(f,"{}",s),
            ApplicationError::InfoSendError(ref e) => write!(f,"{}",e),
            ApplicationError::RecvError(ref e) => write!(f, "{}",e),
            ApplicationError::UsiProtocolError(ref e) => write!(f,"{}",e),
        }
    }
}
impl error::Error for ApplicationError {
    fn description(&self) -> &str {
        match *self {
            ApplicationError::AgentRunningError(_) => "An error occurred while running USIAgent.",
            ApplicationError::LogicError(_) => "Logic error.",
            ApplicationError::InvalidStateError(_) => "Invalid state.",
            ApplicationError::EventDispatchError(_) => "An error occurred while processing the event.",
            ApplicationError::InfoSendError(_) => "An error occurred when sending info command.",
            ApplicationError::RecvError(_) => "An error occurred while receiving the message.",
            ApplicationError::UsiProtocolError(_) => "An error occurred in the parsing or string generation process of string processing according to the USI protocol.",
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            ApplicationError::AgentRunningError(_) => None,
            ApplicationError::LogicError(_) => None,
            ApplicationError::InvalidStateError(_) => None,
            ApplicationError::EventDispatchError(_) => None,
            ApplicationError::InfoSendError(ref e) => Some(e),
            ApplicationError::RecvError(ref e) => Some(e),
            ApplicationError::UsiProtocolError(ref e) => Some(e),
        }
    }
}
impl PlayerError for ApplicationError {}
impl From<UsiProtocolError> for ApplicationError {
    fn from(err: UsiProtocolError) -> ApplicationError {
        ApplicationError::UsiProtocolError(err)
    }
}
impl<'a> From<EventDispatchError<'_, EventQueue<UserEvent, UserEventKind>, UserEvent, ApplicationError>> for ApplicationError {
    fn from(err: EventDispatchError<'_, EventQueue<UserEvent, UserEventKind>, UserEvent, ApplicationError>) -> Self {
        ApplicationError::EventDispatchError(format!("{}",err))
    }
}
impl From<RecvError> for ApplicationError {
    fn from(err: RecvError) -> ApplicationError {
        ApplicationError::RecvError(err)
    }
}
impl From<InfoSendError> for ApplicationError {
    fn from(err: InfoSendError) -> ApplicationError {
        ApplicationError::InfoSendError(err)
    }
}
