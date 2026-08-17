// T6.2: Activation Dump Contract (standalone - no std required)

use core::panic::PanicInfo;

pub struct ActivationDump {
    pub operator_id: &'static str,
    pub layer_id: usize,
}