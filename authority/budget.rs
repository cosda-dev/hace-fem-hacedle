use super::BudgetAction;
use super::BudgetConfig;

pub struct BudgetGuard {
    config: BudgetConfig,
    tokens_used: u32,
    #[cfg(feature = "std")]
    start_time: std::time::Instant,
}

#[cfg(feature = "std")]
impl BudgetGuard {
    pub fn new(config: BudgetConfig) -> Self {
        Self {
            config,
            tokens_used: 0,
            start_time: std::time::Instant::now(),
        }
    }

    pub fn check_token(&self, tokens: u32) -> BudgetAction {
        if self.tokens_used + tokens > self.config.max_tokens {
            BudgetAction::Stop
        } else if self.tokens_used + tokens > self.config.max_tokens / 2 {
            BudgetAction::Throttle
        } else {
            BudgetAction::Continue
        }
    }

    pub fn check_duration(&self) -> BudgetAction {
        let elapsed = self.start_time.elapsed().as_millis() as u64;
        if elapsed > self.config.max_duration_ms {
            BudgetAction::Stop
        } else {
            BudgetAction::Continue
        }
    }
}

#[cfg(not(feature = "std"))]
impl BudgetGuard {
    pub fn new(_config: BudgetConfig) -> Self {
        Self {
            config: _config,
            tokens_used: 0,
        }
    }

    pub fn check_token(&self, _tokens: u32) -> BudgetAction {
        BudgetAction::Continue
    }

    pub fn check_duration(&self) -> BudgetAction {
        BudgetAction::Continue
    }
}

#[cfg(feature = "std")]
impl Default for BudgetGuard {
    fn default() -> Self {
        Self::new(BudgetConfig::default())
    }
}

#[cfg(not(feature = "std"))]
impl Default for BudgetGuard {
    fn default() -> Self {
        Self::new(BudgetConfig::default())
    }
}