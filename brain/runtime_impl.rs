use super::{BrainRuntime, BrainError, ReasonCtx, ReasonResult, TokenStream};

pub struct HacedleBrain {
    session_id: u64,
}

impl HacedleBrain {
    pub fn new() -> Self {
        Self { session_id: 0 }
    }
}

#[cfg(feature = "std")]
#[async_trait::async_trait]
impl BrainRuntime for HacedleBrain {
    fn mount_projection(&self) -> Result<(), BrainError> {
        Ok(())
    }

    fn mount_rules(&self) -> Result<(), BrainError> {
        Ok(())
    }

    fn mount_models(&self) -> Result<(), BrainError> {
        Ok(())
    }

    fn select_provider(&self, _caps: &[&'static str]) -> Option<&'static dyn super::super::x::bridge::Provider> {
        None
    }

    async fn reason(&self, ctx: ReasonCtx) -> Result<ReasonResult, BrainError> {
        Ok(ReasonResult {
            output: ctx.payload,
            confidence: 0.95,
            tokens_used: 0,
            model_id: String::from("hacedle"),
            plan: None,
        })
    }

    async fn stream(&self, _ctx: ReasonCtx) -> Result<TokenStream, BrainError> {
        Ok(TokenStream {
            session_id: self.session_id,
            finished: false,
        })
    }
}

impl Default for HacedleBrain {
    fn default() -> Self {
        Self::new()
    }
}