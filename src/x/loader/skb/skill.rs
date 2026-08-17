use alloc::string::String;

pub struct SkillLayer {
    pub prompts: String,
    pub templates: String,
}

impl SkillLayer {
    pub fn new() -> Self {
        Self {
            prompts: String::new(),
            templates: String::new(),
        }
    }
}

impl Default for SkillLayer {
    fn default() -> Self {
        Self::new()
    }
}
